import json
import logging
import os
import random
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from database.util import DataElement, _evaluate_loss, _evaluate_result, log_agent_run

class CandidateManager:
    """Candidate set manager, maintains top-50 data elements."""

    def __init__(self,
                 capacity: int = 50,
                 update_threshold: int = 50,
                 storage_file: str = "candidate_storage.json",
                 get_element_by_index_func: Optional[Callable[[int], DataElement]] = None,
                 update_element_score_func: Optional[Callable[[int, float], bool]] = None):
        """
        Initialize candidate set manager.

        Args:
            capacity: Candidate set capacity
            update_threshold: Threshold for triggering updates based on new data count
            storage_file: Storage file path
            get_element_by_index_func: Callback function to get element by index
            update_element_score_func: Callback function to update element score
        """
        self.capacity = capacity
        self.update_threshold = update_threshold
        self.storage_file = storage_file
        self.lock = threading.RLock()

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Candidate set data: List[DataElement]
        self.candidates: List[DataElement] = []

        # Counter: number of newly added data
        self.new_data_count = 0

        # Dependency injection callback functions
        self._get_element_by_index = get_element_by_index_func
        self._update_element_score = update_element_score_func

        # Load saved candidate set
        self._load_candidates()

    def set_database_callbacks(self,
                              get_element_by_index_func: Callable[[int], DataElement],
                              update_element_score_func: Callable[[int, float], bool]):
        """
        Set database callback functions (dependency injection).

        Args:
            get_element_by_index_func: Function to get element by index
            update_element_score_func: Function to update element score
        """
        self._get_element_by_index = get_element_by_index_func
        self._update_element_score = update_element_score_func

    def _safe_create_dataelement_from_dict(self, element_dict: Dict[str, Any]) -> DataElement:
        """Safely create DataElement from dictionary, handling unknown fields"""
        try:
            return DataElement.from_dict(element_dict)
        except TypeError as e:
            self.logger.debug(f"Filtering unknown fields from DataElement creation: {e}")
            
            # Define known fields that DataElement accepts
            known_fields = {
                'time', 'name', 'result', 'program', 'analysis', 
                'cognition', 'log', 'motivation', 'index', 'parent', 
                'summary', 'score', 'motivation_embedding',
                # Include new fields
                'name_new', 'parameters', 'svg_picture'
            }
            
            # Filter dictionary to only include known fields
            filtered_dict = {k: v for k, v in element_dict.items() if k in known_fields}
            
            # Ensure required fields have defaults
            defaults = {
                'time': '',
                'name': '',
                'result': {},
                'program': '',
                'analysis': '',
                'cognition': '',
                'log': '',
                'motivation': '',
                'index': 0,
                'parent': None,
                'summary': '',
                'score': None,
                'name_new': '',
                'parameters': '',
                'svg_picture': ''
            }
            
            for field, default in defaults.items():
                if field not in filtered_dict:
                    filtered_dict[field] = default
            
            return DataElement.from_dict(filtered_dict)

    async def _evaluate_element(self, element: DataElement) -> float:
        """
        Evaluate data element score.

        Args:
            element: Data element

        Returns:
            float: Score
        """
        import math

        # 1. Calculate loss score (normalized to 10 points)
        try:
            loss = _evaluate_loss(element.result)
            # Baseline value
            base_loss = 4.575
            # Calculate position relative to 10% range (smaller loss is better, so use base_loss - loss)
            loss_relative = (base_loss - loss) / (base_loss * 0.1)  # Range [-1,+1] corresponds to Â±10%
            # Map to [-6, +6] for sigmoid transformation (fully utilize sigmoid response range)
            loss_clamped = max(-1, min(1, loss_relative))
            loss_score = 1 / (1 + math.exp(-6 * loss_clamped))
        except Exception as e:
            self.logger.warning(f"Failed to calculate loss score: {e}")
            loss_score = 0.5  # Default score (middle value of 0-1 range)

        # 2. Calculate benchmark score (normalized to 10 points)
        try:
            benchmark = _evaluate_result(element.result)
            # Baseline value
            base_benchmark = 0.224
            # Calculate position relative to 10% range (larger benchmark is better)
            benchmark_relative = (benchmark - base_benchmark) / (base_benchmark * 0.1)
            # Map to [-6, +6] for sigmoid transformation
            benchmark_clamped = max(-1, min(1, benchmark_relative))
            benchmark_score = 1 / (1 + math.exp(-6 * benchmark_clamped))
        except Exception as e:
            self.logger.warning(f"Failed to calculate benchmark score: {e}")
            benchmark_score = 0.5  # Default score (middle value of 0-1 range)

        # 3. Call model judger agent to get score (needs normalization to 0-1)
        try:
            # Fix import path
            from evaluate_agent.model import model_judger
            from evaluate_agent.prompt import model_judger_input
            from agents import set_default_openai_client, set_default_openai_api, set_tracing_disabled
            from openai import AsyncAzureOpenAI

            client = AsyncAzureOpenAI()
            set_default_openai_client(client)
            set_default_openai_api("chat_completions")
            set_tracing_disabled(True)

            input_text = model_judger_input(element)
            result = await log_agent_run("Model Judger", model_judger, input_data=input_text)
            agent_score_raw = float(result.final_output.weighted_final_score)  # Agent returns score 1-10

            # Normalize 1-10 score to 0-1 range to match loss_score and benchmark_score
            agent_score = (agent_score_raw - 1) / 9  # Map 1-10 to 0-1

        except Exception as e:
            self.logger.warning(f"Failed to call model judger: {e}")
            agent_score = 0.5  # Default score adjusted to 0.5 (middle value of 0-1 range)

        total_score = loss_score + benchmark_score + agent_score

        self.logger.info(f"Element {element.index} score details: loss_score={loss_score:.3f}(0-1), benchmark_score={benchmark_score:.3f}(0-1), agent_score={agent_score:.3f}(0-1), total={total_score:.3f}")

        # Update in-memory object and database score cache
        element.score = total_score
        if self._update_element_score:
            self.logger.info(f"Saving score to database for element {element.index}")
            self._update_element_score(element.index, total_score)

        return total_score

    def _evaluate_filter(self, element: DataElement, standard_element: Optional[DataElement] = None) -> bool:
        """
        Evaluate data element filter conditions.

        Args:
            element: Data element
            standard_element: Standard reference element (optional)
        """
        if standard_element is None:
            # If no standard element or cannot get one, simplify filtering
            return True

        try:
            loss = _evaluate_loss(element.result)
            benchmark = _evaluate_result(element.result)
            standard_loss = _evaluate_loss(standard_element.result)
            standard_benchmark = _evaluate_result(standard_element.result)

            if standard_loss == 0 or standard_benchmark == 0:
                return True  # Avoid division by zero error

            loss_delta = abs(loss - standard_loss) / standard_loss
            benchmark_delta = (benchmark - standard_benchmark) / standard_benchmark

            # Allow loss_delta within 0.1 range, but no upper limit for benchmark
            if loss_delta < 0.1 and benchmark_delta > -0.1:
                return True
            else:
                self.logger.info(f"Element {element.index} did not pass filter conditions")
                return False

        except Exception as e:
            self.logger.warning(f"Filter condition evaluation failed: {e}")
            return True  # Allow pass when evaluation fails

    def _get_standard_element(self) -> Optional[DataElement]:
        """
        Get standard reference element.

        Returns:
            Optional[DataElement]: Standard element, returns None if retrieval fails
        """
        if self._get_element_by_index is None:
            self.logger.warning("Database callback function not set, cannot get standard element")
            return None

        try:
            return self._get_element_by_index(1)
        except Exception as e:
            self.logger.warning(f"Failed to get standard element: {e}")
            return None

    def _load_candidates(self):
        """Load candidate set from file."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                candidates_data = data.get('candidates', [])
                self.new_data_count = data.get('new_data_count', 0)

                # Rebuild candidate set
                self.candidates = []
                for item in candidates_data:
                    element_dict = item
                    # Compatible with old format: {'element': {...}, 'score': ...}
                    if 'element' in item and isinstance(item['element'], dict):
                        element_dict = item['element']
                        if 'score' not in element_dict and 'score' in item:
                            element_dict['score'] = item.get('score')

                    # Use safe creation method
                    element = self._safe_create_dataelement_from_dict(element_dict)
                    self.candidates.append(element)

                # Re-sort to ensure consistency
                self.candidates.sort(key=lambda x: x.score if x.score is not None else -1, reverse=True)

                self.logger.info(f"Successfully loaded {len(self.candidates)} candidates (compatible with old format)")
            else:
                self.logger.info("Candidate storage file not found, creating new candidate set")

        except Exception as e:
            self.logger.error(f"Failed to load candidate set: {e}")
            self.candidates = []
            self.new_data_count = 0

    def _save_candidates(self):
        """Save candidate set to file."""
        try:
            candidates_to_save = []
            for element in self.candidates:
                element_dict = element.to_dict()
                element_dict.pop('motivation_embedding', None)  # Remove embedding
                candidates_to_save.append(element_dict)

            data = {
                'candidates': candidates_to_save,
                'new_data_count': self.new_data_count,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.debug("Candidate set saved successfully")

        except Exception as e:
            self.logger.error(f"Failed to save candidate set: {e}")

    def increment_count(self) -> bool:
        """
        Increment new data count, check if update should be triggered.

        Returns:
            bool: Whether candidate set update should be triggered
        """
        with self.lock:
            try:
                # Increment new data count
                self.new_data_count += 1

                # Check if batch update should be triggered
                if self.new_data_count >= self.update_threshold:
                    self.logger.info(f"Reached update threshold {self.update_threshold}, candidate set update needed")
                    return True

                return False

            except Exception as e:
                self.logger.error(f"Candidate set counting failed: {e}")
                return False

    async def add_element(self, element: DataElement) -> bool:
        """
        Directly add element to candidate set (for manual addition).

        Args:
            element: Data element

        Returns:
            bool: Whether addition was successful
        """
        with self.lock:
            try:
                # Calculate element score
                score = await self._evaluate_element(element)

                # If candidate set is not full, add directly
                if len(self.candidates) < self.capacity:
                    self.candidates.append(element)
                    self.candidates.sort(key=lambda x: x.score, reverse=True)
                    self._save_candidates()
                    self.logger.info(f"Successfully added element to candidate set: index={element.index}, score={score:.4f}")
                    return True

                # Otherwise check if it can replace the lowest scoring candidate
                elif score > self.candidates[-1].score:
                    old_score = self.candidates[-1].score
                    self.candidates[-1] = element
                    self.candidates.sort(key=lambda x: x.score, reverse=True)
                    self._save_candidates()
                    self.logger.info(f"Replaced lowest scoring candidate: index={element.index}, new score={score:.4f}, old score={old_score:.4f}")
                    return True

                else:
                    self.logger.info(f"Element score too low, not added to candidate set: index={element.index}, score={score:.4f}, lowest score={self.candidates[-1].score:.4f}")
                    return False

            except Exception as e:
                self.logger.error(f"Failed to add element to candidate set: {e}")
                return False

    async def update_candidates(self, new_elements: List[DataElement]) -> Dict[str, Any]:
        """
        Update candidate set: compare newly added data with existing candidates, find top-50.

        Args:
            new_elements: List of newly added data elements

        Returns:
            Dict[str, Any]: Update statistics
        """
        with self.lock:
            try:
                self.logger.info(f"Starting candidate set update, current candidates: {len(self.candidates)}, new data: {len(new_elements)}")

                # Merge existing candidates and new data
                all_elements: List[DataElement] = self.candidates.copy()

                standard_element = self._get_standard_element()
                filtered_elements = []

                if standard_element is None:
                    self.logger.warning("Standard element not set, cannot perform filtering")
                else:
                    for element in new_elements:
                        if not self._evaluate_filter(element, standard_element):
                            self.logger.info(f"Element {element.index} did not pass filter conditions")
                            continue
                        filtered_elements.append(element)

                # Evaluate and add new data
                new_evaluated = 0
                for element in filtered_elements:
                    await self._evaluate_element(element)  # Score will be saved in element.score
                    all_elements.append(element)
                    new_evaluated += 1

                # Sort by score, take top-capacity
                all_elements.sort(key=lambda x: x.score, reverse=True)
                old_count = len(self.candidates)
                old_candidates = self.candidates.copy()
                self.candidates = all_elements[:self.capacity]

                # Reset counter
                self.new_data_count = 0

                # Save updated candidate set
                self._save_candidates()

                # Calculate change statistics
                new_entries = 0
                replaced_entries = 0
                old_indices = {el.index for el in old_candidates}

                for element in self.candidates:
                    if element.index not in old_indices:
                        new_entries += 1

                replaced_entries = old_count + new_evaluated - len(self.candidates) - (len(all_elements) - len(self.candidates))

                # Statistics
                stats = {
                    "updated_at": datetime.now().isoformat(),
                    "total_evaluated": len(all_elements),
                    "new_elements_evaluated": new_evaluated,
                    "old_candidates": old_count,
                    "final_candidates": len(self.candidates),
                    "new_entries": new_entries,
                    "highest_score": self.candidates[0].score if self.candidates else 0,
                    "lowest_score": self.candidates[-1].score if self.candidates else 0,
                    "replaced_entries": replaced_entries,
                    "average_score": sum(el.score for el in self.candidates) / len(self.candidates) if self.candidates else 0
                }

                self.logger.info(f"Candidate set update completed: evaluated {new_evaluated} new elements, added {new_entries} new elements to candidate set")

                return stats

            except Exception as e:
                self.logger.error(f"Failed to update candidate set: {e}")
                return {"error": str(e)}

    def replace_candidates(self, new_candidates: List[DataElement]) -> Dict[str, Any]:
        """
        Completely replace existing candidate set with new element list.

        Args:
            new_candidates: New candidate element list (should be sorted by score)

        Returns:
            Dict[str, Any]: Update statistics
        """
        with self.lock:
            try:
                old_count = len(self.candidates)

                # Ensure new candidates don't exceed capacity
                self.candidates = new_candidates[:self.capacity]

                # Save updated candidate set
                self._save_candidates()

                new_count = len(self.candidates)

                stats = {
                    "updated_at": datetime.now().isoformat(),
                    "old_candidate_count": old_count,
                    "new_candidate_count": new_count,
                    "message": f"Candidate list replaced with {new_count} elements."
                }

                self.logger.info(f"Candidate set completely replaced. Old count: {old_count}, New count: {new_count}")

                return stats

            except Exception as e:
                self.logger.error(f"Failed to replace candidate set: {e}")
                return {"error": str(e)}

    def get_top_k(self, k: int = 10) -> List[DataElement]:
        """
        Get top-k candidates.

        Args:
            k: Number to return

        Returns:
            List[DataElement]: top-k data elements
        """
        with self.lock:
            try:
                k = min(k, len(self.candidates))
                return self.candidates[:k]
            except Exception as e:
                self.logger.error(f"Failed to get top-k: {e}")
                return []

    def get_all_candidates(self) -> List[DataElement]:
        """
        Get all candidates and their scores.

        Returns:
            List[DataElement]: Candidate list
        """
        with self.lock:
            try:
                # Return copy to avoid external modification
                return self.candidates.copy()
            except Exception as e:
                self.logger.error(f"Failed to get all candidates: {e}")
                return []

    def sample_from_range(self, a: int, b: int, k: int) -> List[DataElement]:
        """
        Randomly sample k candidates from the a-th to b-th position range after sorting.

        Args:
            a: Range start position (1-based counting)
            b: Range end position (1-based counting, inclusive)
            k: Sample count

        Returns:
            List[DataElement]: k randomly sampled elements
        """
        with self.lock:
            try:
                # Parameter validation
                if a < 1 or b < 1:
                    self.logger.error("Range positions must start from 1")
                    return []

                if a > b:
                    self.logger.error("Start position cannot be greater than end position")
                    return []

                if k < 1:
                    self.logger.error("Sample count must be greater than 0")
                    return []

                total_count = len(self.candidates)
                if a > total_count:
                    self.logger.warning(f"Start position {a} exceeds candidate set size {total_count}")
                    return []

                # Adjust end position
                actual_b = min(b, total_count)

                # Extract candidates from specified range (convert to 0-based indexing)
                range_candidates = self.candidates[a-1:actual_b]
                range_size = len(range_candidates)

                # If range has fewer elements than k, return all available elements
                if range_size <= k:
                    return range_candidates

                # Randomly sample k elements
                return random.sample(range_candidates, k)

            except Exception as e:
                self.logger.error(f"Range random sampling failed: {e}")
                return []

    def delete_by_index(self, index: int) -> bool:
        """
        Delete candidate by index.

        Args:
            index: Data element index

        Returns:
            bool: Whether deletion was successful
        """
        with self.lock:
            try:
                # Find and delete matching candidate
                for i, element in enumerate(self.candidates):
                    if element.index == index:
                        del self.candidates[i]
                        self._save_candidates()
                        self.logger.info(f"Successfully deleted element with index={index} from candidate set")
                        return True

                self.logger.warning(f"Element with index={index} not found in candidate set")
                return False

            except Exception as e:
                self.logger.error(f"Failed to delete element from candidate set: {e}")
                return False

    def delete_by_name(self, name: str) -> int:
        """
        Delete candidates by name.

        Args:
            name: Data element name

        Returns:
            int: Number of deletions
        """
        with self.lock:
            try:
                deleted_count = 0
                i = 0
                while i < len(self.candidates):
                    element = self.candidates[i]
                    if element.name == name:
                        del self.candidates[i]
                        deleted_count += 1
                    else:
                        i += 1

                if deleted_count > 0:
                    self._save_candidates()
                    self.logger.info(f"Successfully deleted {deleted_count} elements with name={name} from candidate set")

                return deleted_count

            except Exception as e:
                self.logger.error(f"Failed to delete elements from candidate set: {e}")
                return 0

    async def update_element(self, element: DataElement) -> bool:
        """
        Update element in candidate set.

        Args:
            element: Updated data element

        Returns:
            bool: Whether update was successful
        """
        with self.lock:
            try:
                # Find and update matching candidate
                for i, old_element in enumerate(self.candidates):
                    if old_element.index == element.index:
                        await self._evaluate_element(element)  # Will update element.score
                        self.candidates[i] = element

                        # Re-sort
                        self.candidates.sort(key=lambda x: x.score, reverse=True)

                        self._save_candidates()
                        self.logger.info(f"Successfully updated element with index={element.index} in candidate set, score changed from {old_element.score:.4f} to {element.score:.4f}")
                        return True

                self.logger.warning(f"Element with index={element.index} not found in candidate set")
                return False

            except Exception as e:
                self.logger.error(f"Failed to update candidate set element: {e}")
                return False

    def get_stats(self) -> Dict[str, Any]:
        """Get candidate set statistics."""
        with self.lock:
            try:
                if not self.candidates:
                    return {
                        "capacity": self.capacity,
                        "current_size": 0,
                        "update_threshold": self.update_threshold,
                        "new_data_count": self.new_data_count,
                        "needs_update": self.new_data_count >= self.update_threshold
                    }

                scores = [el.score for el in self.candidates]

                return {
                    "capacity": self.capacity,
                    "current_size": len(self.candidates),
                    "update_threshold": self.update_threshold,
                    "new_data_count": self.new_data_count,
                    "needs_update": self.new_data_count >= self.update_threshold,
                    "highest_score": max(scores),
                    "lowest_score": min(scores),
                    "average_score": sum(scores) / len(scores),
                    "score_range": max(scores) - min(scores)
                }

            except Exception as e:
                self.logger.error(f"Failed to get candidate set statistics: {e}")
                return {"error": str(e)}

    def get_new_data_count(self) -> int:
        """Get current new data count."""
        with self.lock:
            return self.new_data_count

    def clear(self) -> bool:
        """Clear candidate set."""
        with self.lock:
            try:
                self.candidates = []
                self.new_data_count = 0
                self._save_candidates()
                self.logger.info("Candidate set cleared")
                return True
            except Exception as e:
                self.logger.error(f"Failed to clear candidate set: {e}")
                return False


# Global candidate manager instance
_candidate_manager = None

def get_candidate_manager() -> CandidateManager:
    """Get global candidate manager instance."""
    global _candidate_manager
    if _candidate_manager is None:
        _candidate_manager = CandidateManager()
    return _candidate_manager
