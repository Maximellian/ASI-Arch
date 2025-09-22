import os
import logging
import threading
import csv
import io
from typing import Dict, List, Optional, Any
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, DuplicateKeyError, PyMongoError
from database.util import DataElement
from database.embedding_service import get_embedding_service
from database.faiss_manager import get_faiss_manager
from database.candidate_manager import get_candidate_manager

class MongoDatabase:
    """Persistent MongoDB database"""

    def __init__(
        self,
        connection_string: str = None,
        database_name: str = "myapp",
        collection_name: str = "data_elements",
        username: str = None,
        password: str = None,
    ):
        # Use environment variable if no connection_string provided
        if connection_string is None:
            connection_string = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017')
    
        self.connection_string = connection_string
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.lock = threading.RLock()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self._initialize_connection(username, password)
        self._setup_candidate_manager_callbacks()

    def _initialize_connection(self, username: str = None, password: str = None):
        try:
            if username and password:
                if "://" in self.connection_string:
                    protocol, rest = self.connection_string.split("://", 1)
                    if "@" in rest:
                        connection_url = self.connection_string
                    else:
                        host_port = rest.split("/")[0]
                        db_part = "/" + "/".join(rest.split("/")[1:]) if "/" in rest else ""
                        connection_url = f"{protocol}://{username}:{password}@{host_port}{db_part}"
                else:
                    connection_url = self.connection_string
            else:
                connection_url = self.connection_string

            self.client = MongoClient(
                connection_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
            )
            self.client.admin.command("ping")
            self.logger.info("MongoDB connection successful")
            self.db: Database = self.client[self.database_name]
            self.collection: Collection = self.db[self.collection_name]
            self._create_indexes()

        except ConnectionFailure as e:
            self.logger.error(f"MongoDB connection failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"MongoDB initialization failed: {e}")
            raise

    def _setup_candidate_manager_callbacks(self):
        try:
            candidate_manager = get_candidate_manager()
            candidate_manager.set_database_callbacks(
                get_element_by_index_func=self.get_by_index,
                update_element_score_func=self.update_element_score
            )
            self.logger.info("Candidate manager callbacks set up")
        except Exception as e:
            self.logger.warning(f"Failed to set candidate manager callbacks: {e}")

    def _create_indexes(self):
        try:
            self.collection.create_index([("name", ASCENDING)])
            self.collection.create_index([("time", ASCENDING)])
            self.collection.create_index([("index", ASCENDING)])
            self.collection.create_index([("parent", ASCENDING)])
            self.collection.create_index([("name", ASCENDING), ("time", ASCENDING)])
            self.collection.create_index([("index", ASCENDING), ("name", ASCENDING)])
            self.collection.create_index([("parent", ASCENDING), ("index", ASCENDING)])
            self.logger.info("Indexes created")
        except Exception as e:
            self.logger.warning(f"Failed to create indexes: {e}")

    def _safe_create_dataelement(self, item_dict):
        """Safely create DataElement from dictionary, handling unknown fields."""
        if not isinstance(item_dict, dict):
            try:
                item_dict = vars(item_dict)
            except Exception:
                self.logger.error(f"_safe_create_dataelement: Failed to convert item to dict: {type(item_dict)}")
                raise
        try:
            return DataElement(**item_dict)
        except TypeError as e:
            self.logger.debug(f"Filtering unknown fields from DataElement creation: {e}")
            known_fields = {
                'time', 'name', 'result', 'program', 'analysis', 
                'cognition', 'log', 'motivation', 'index', 'parent', 'summary',
                'name_new', 'parameters', 'svg_picture'
            }
            filtered_dict = {k: v for k, v in item_dict.items() if k in known_fields}
            if 'motivation_embedding' in item_dict:
                filtered_dict['motivation_embedding'] = item_dict['motivation_embedding']
            if 'score' in item_dict:
                filtered_dict['score'] = item_dict['score']
            defaults = {
                'time': 'unknown',
                'name': 'unknown',
                'result': {},
                'program': '',
                'analysis': '',
                'cognition': '',
                'log': '',
                'motivation': '',
                'index': 0,
                'parent': None,
                'summary': '',
                'name_new': '',
                'parameters': '',
                'svg_picture': ''
            }
            for field, default in defaults.items():
                if field not in filtered_dict:
                    filtered_dict[field] = default
            return DataElement(**filtered_dict)

    def _validate_element(self, element: DataElement) -> bool:
        """Validates a data element"""
        # Validate required string fields
        required_str_fields = ['name', 'time', 'program', 'analysis', 'cognition', 'log', 'motivation']
        for field in required_str_fields:
            field_value = getattr(element, field, None)
            if not field_value or not isinstance(field_value, str):
                self.logger.error(f"Data validation failed: {field} field is empty or of incorrect type")
                return False

        # Validate optional string fields (can be empty but must be strings)
        optional_str_fields = ['summary', 'name_new', 'parameters', 'svg_picture']
        for field in optional_str_fields:
            field_value = getattr(element, field, '')
            if not isinstance(field_value, str):
                self.logger.error(f"Data validation failed: {field} field has incorrect type (should be a string)")
                return False

        if not element.result or not isinstance(element.result, dict):
            self.logger.error("Data validation failed: result field is empty or of incorrect type (should be a dictionary)")
            return False

        if not isinstance(element.index, int):
            self.logger.error("Data validation failed: index field has incorrect type")
            return False

        # Validate 'parent' field: can be None or a positive integer
        if element.parent is not None and (not isinstance(element.parent, int) or element.parent <= 0):
            self.logger.error("Data validation failed: parent field must be None or a positive integer")
            return False

        return True

    def _get_next_index(self) -> int:
        """Gets the next index value"""
        try:
            # Find the current largest index value
            cursor = self.collection.find({}).sort("index", -1).limit(1)
            docs = list(cursor)
            if docs:
                return docs[0].get("index", 0) + 1
            else:
                return 1
        except Exception as e:
            self.logger.warning(f"Failed to get next index, using default value: {e}")
            return 1

    async def add_element(self, time: str, name: str, result: Dict[str, Any], program: str,
                         analysis: str, cognition: str, log: str, motivation: str, 
                         parent: Optional[int] = None, summary: str = "",
                         name_new: str = "", parameters: str = "", svg_picture: str = "") -> bool:
        """Adds a data element"""
        with self.lock:
            try:
                # Get the next index
                next_index = self._get_next_index()

                # Handle null values
                time = time or "none"
                name = name or "none"
                result = result or {}
                program = program or "none"
                analysis = analysis or "none"
                cognition = cognition or "none"
                log = log or "none"
                summary = summary or ""
                name_new = name_new or ""
                parameters = parameters or ""
                svg_picture = svg_picture or ""

                # Calculate motivation embedding
                motivation_embedding = None
                if motivation and motivation != "none":
                    try:
                        self.logger.info(f"Starting to calculate motivation embedding, length: {len(motivation)}")
                        embedding_service = get_embedding_service()
                        motivation_embedding = embedding_service.get_single_embedding(motivation)
                        self.logger.info(f"Successfully calculated motivation embedding, dimension: {len(motivation_embedding)}")
                    except Exception as e:
                        self.logger.error(f"Failed to calculate motivation embedding: {e}", exc_info=True)
                        motivation_embedding = None
                else:
                    self.logger.info(f"Motivation is empty or 'none', skipping embedding calculation")

                if not motivation:
                    motivation = "none"

                # Validate parent existence (if specified)
                if parent is not None:
                    parent_element = self.get_by_index(parent)
                    if parent_element is None:
                        self.logger.error(f"The specified parent index={parent} does not exist")
                        return False

                # Create the data element
                element = DataElement(
                    time=time,
                    name=name,
                    result=result,
                    program=program,
                    analysis=analysis,
                    cognition=cognition,
                    log=log,
                    motivation=motivation,
                    index=next_index,
                    motivation_embedding=motivation_embedding,
                    parent=parent,
                    summary=summary,
                    name_new=name_new,
                    parameters=parameters,
                    svg_picture=svg_picture
                )

                # Validate the data
                if not self._validate_element(element):
                    return False

                # Convert to dictionary and add timestamp
                doc = element.to_dict()
                doc['created_at'] = datetime.now()
                doc['updated_at'] = datetime.now()

                # Insert into MongoDB
                result = self.collection.insert_one(doc)
                if result.inserted_id:
                    # Candidate set count management
                    candidate_manager = get_candidate_manager()
                    self.logger.info(f"Successfully obtained candidate set manager, current count: {candidate_manager.new_data_count}")
                    should_update = candidate_manager.increment_count()
                    self.logger.info(f"Candidate set count increased to: {candidate_manager.new_data_count}, needs update: {should_update}")

                    # If the candidate set needs to be updated, get the latest 50 data entries for batch update
                    if should_update:
                        self.logger.info("Reached the candidate set update threshold, starting batch update")
                        # Get the 50 most recently added data for candidate set update
                        new_elements = self._get_recent_elements(50)
                        self.logger.info(f"Retrieved {len(new_elements)} latest elements for candidate set update")
                        update_stats = await candidate_manager.update_candidates(new_elements)
                        self.logger.info(f"Candidate set update completed: {update_stats}")

                    self.logger.info(f"Successfully added element: {name}, ID: {result.inserted_id}, Index: {next_index}")

                    # If there is an embedding, add it to the FAISS index
                    if motivation_embedding is not None and len(motivation_embedding) > 0:
                        self.logger.info(f"Preparing to add vector to FAISS, vector dimension: {len(motivation_embedding)}")
                        faiss_manager = get_faiss_manager()
                        faiss_manager.add_vector(motivation_embedding, str(result.inserted_id))
                        self.logger.info(f"Vector successfully added to FAISS index: {result.inserted_id}")
                    else:
                        self.logger.info("No valid embedding, skipping FAISS index addition")

                    return True
                else:
                    self.logger.error("Insertion failed, no ID returned")
                    return False

            except DuplicateKeyError:
                self.logger.error("Data is duplicated")
                return False
            except PyMongoError as e:
                self.logger.error(f"MongoDB operation failed: {e}")
                return False
            except Exception as e:
                self.logger.error(f"Failed to add element: {e}")
                return False

    def delete_element_by_name(self, name: str) -> bool:
        """Deletes a data element by name"""
        with self.lock:
            try:
                # First, find the document to be deleted to get its _id
                doc_to_delete = self.collection.find_one({"name": name})
                if not doc_to_delete:
                    self.logger.warning(f"Element with name={name} not found")
                    return False

                doc_id = str(doc_to_delete["_id"])

                # Delete from MongoDB
                result = self.collection.delete_one({"name": name})
                if result.deleted_count > 0:
                    self.logger.info(f"Successfully deleted element with name={name} from MongoDB")

                    # If there is an embedding, delete it from FAISS
                    if doc_to_delete.get("motivation_embedding"):
                        try:
                            faiss_manager = get_faiss_manager()
                            faiss_manager.remove_vector(doc_id)
                            self.logger.info(f"Successfully removed vector from FAISS, ID: {doc_id}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove vector from FAISS: {e}")

                    # Delete from the candidate set
                    try:
                        candidate_manager = get_candidate_manager()
                        candidate_manager.delete_by_name(name)
                    except Exception as e:
                        self.logger.warning(f"Failed to delete from candidate set: {e}")

                    return True
                else:
                    self.logger.error(f"Failed to delete element with name={name} from MongoDB")
                    return False

            except PyMongoError as e:
                self.logger.error(f"MongoDB error occurred while deleting element: {e}")
                return False
            except Exception as e:
                self.logger.error(f"An unknown error occurred while deleting element: {e}")
                return False

    def delete_element_by_index(self, index: int) -> bool:
        """Deletes a data element by index"""
        with self.lock:
            try:
                # First, find the document to be deleted to get its _id
                doc_to_delete = self.collection.find_one({"index": index})
                if not doc_to_delete:
                    self.logger.warning(f"Element with index={index} not found")
                    return False

                doc_id = str(doc_to_delete["_id"])

                # Delete from MongoDB
                result = self.collection.delete_one({"index": index})
                if result.deleted_count > 0:
                    self.logger.info(f"Successfully deleted element with index={index} from MongoDB")

                    # If there is an embedding, delete it from FAISS
                    if doc_to_delete.get("motivation_embedding"):
                        try:
                            faiss_manager = get_faiss_manager()
                            faiss_manager.remove_vector(doc_id)
                            self.logger.info(f"Successfully removed vector from FAISS, ID: {doc_id}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove vector from FAISS: {e}")

                    # Delete from the candidate set
                    try:
                        candidate_manager = get_candidate_manager()
                        candidate_manager.delete_by_index(index)
                    except Exception as e:
                        self.logger.warning(f"Failed to delete from candidate set: {e}")

                    return True
                else:
                    self.logger.error(f"Failed to delete element with index={index} from MongoDB")
                    return False

            except PyMongoError as e:
                self.logger.error(f"MongoDB error occurred while deleting element: {e}")
                return False
            except Exception as e:
                self.logger.error(f"An unknown error occurred while deleting element: {e}")
                return False

    def delete_all_elements(self) -> bool:
        """Deletes all data elements"""
        with self.lock:
            try:
                # Get all _id's of documents for FAISS deletion
                docs_with_embedding = list(self.collection.find(
                    {"motivation_embedding": {"$exists": True, "$ne": None}},
                    {"_id": 1}
                ))

                # Clear the MongoDB collection
                result = self.collection.delete_many({})
                self.logger.info(f"Successfully deleted {result.deleted_count} elements from MongoDB")

                # If there is FAISS data, clear the FAISS index
                if docs_with_embedding:
                    try:
                        faiss_manager = get_faiss_manager()
                        for doc in docs_with_embedding:
                            doc_id = str(doc["_id"])
                            faiss_manager.remove_vector(doc_id)
                        self.logger.info(f"Successfully removed {len(docs_with_embedding)} vectors from FAISS")
                    except Exception as e:
                        self.logger.warning(f"Failed to clear vectors from FAISS: {e}")

                # Clear the candidate set
                try:
                    candidate_manager = get_candidate_manager()
                    candidate_manager.clear()
                    self.logger.info("Successfully cleared candidate set")
                except Exception as e:
                    self.logger.warning(f"Failed to clear candidate set: {e}")

                return True

            except PyMongoError as e:
                self.logger.error(f"MongoDB error occurred while deleting all elements: {e}")
                return False
            except Exception as e:
                self.logger.error(f"An unknown error occurred while deleting all elements: {e}")
                return False

    def sample_element(self) -> Optional[DataElement]:
        """Samples a random element"""
        with self.lock:
            try:
                # Use MongoDB's $sample aggregation pipeline for random sampling
                pipeline = [{"$sample": {"size": 1}}]
                cursor = self.collection.aggregate(pipeline)
                docs = list(cursor)

                if not docs:
                    return None

                doc = docs[0]
                # Remove MongoDB-specific fields
                doc.pop('_id', None)
                doc.pop('created_at', None)
                doc.pop('updated_at', None)

                return self._safe_create_dataelement(doc)

            except PyMongoError as e:
                self.logger.error(f"Sampling failed: {e}")
                return None
            except Exception as e:
                self.logger.error(f"Sampling failed: {e}")
                return None

    def get_by_name(self, name: str) -> List[DataElement]:
        """Gets all elements matching the given name"""
        with self.lock:
            try:
                # Query for all documents matching the name
                cursor = self.collection.find({"name": name}).sort("time", ASCENDING)
                elements = []

                for doc in cursor:
                    # Remove MongoDB-specific fields
                    doc.pop('_id', None)
                    doc.pop('created_at', None)
                    doc.pop('updated_at', None)

                    element = self._safe_create_dataelement(doc)
                    elements.append(element)

                return elements

            except PyMongoError as e:
                self.logger.error(f"Failed to get elements by name: {e}")
                return []
            except Exception as e:
                self.logger.error(f"Failed to get elements by name: {e}")
                return []

    def get_by_index(self, index: int) -> DataElement:
        """Gets an element by its index"""
        with self.lock:
            try:
                # Query for the document matching the index
                cursor = self.collection.find({"index": index}).sort("time", ASCENDING)
                docs = list(cursor)

                if not docs:
                    return None

                doc = docs[0]
                # Remove MongoDB-specific fields
                doc.pop('_id', None)
                doc.pop('created_at', None)
                doc.pop('updated_at', None)

                return self._safe_create_dataelement(doc)

            except PyMongoError as e:
                self.logger.error(f"Failed to get element by index: {e}")
                return None
            except Exception as e:
                self.logger.error(f"Failed to get element by index: {e}")
                return None

    def _evaluate_result(self, result: dict) -> float:
        """
        Evaluates the result string (CSV format) and returns the average score of all numerical values.
        Skips the header, calculates the mean of the first data row's values.
        """
        if not result.get('test'):
            self.logger.warning("benchmark result is an empty string")
            return 0.0

        try:
            # First calculate test_score
            f = io.StringIO(result['test'])
            reader = csv.reader(f)
            # Skip the header row
            header = next(reader)
            # Read the first data row
            values_list = next(reader)
            scores = []

            # From the second column (skip the model name column)
            for value in values_list[1:]:  # Skip the first column i.e. model name
                if not value.strip():
                    continue
                try:
                    scores.append(float(value))
                except (ValueError, TypeError):
                    self.logger.warning(f"Cannot convert '{value}' to float, ignored when calculating the mean")

            if not scores:
                self.logger.warning(f"No valid numerical values found in the result string: '{result}'")
                return 0.0

            return sum(scores) / len(scores)

        except StopIteration:
            self.logger.warning("CSV data incomplete, missing data row")
            return 0.0
        except Exception as e:
            self.logger.error(f"Error processing CSV data: {e}")
            return 0.0

    def _evaluate_loss(self, result: dict) -> float:
        if not result['train']:
            self.logger.warning("training loss result is empty string")
            return 0.0

        try:
            f = io.StringIO(result['train'])
            reader = csv.reader(f)
            # Skip the header row
            header = next(reader)
            # Read the first data row
            values_list = next(reader)

            # Get last value
            last_value = values_list[-1]  # Last step's loss

            if not last_value.strip():
                self.logger.warning("last step loss value is empty")
                return 0.0

            try:
                return float(last_value)
            except (ValueError, TypeError):
                self.logger.warning(f"Failed to convert the loss value '{last_value}' of the last step to float")
                return 0.0

        except StopIteration:
            self.logger.warning("CSV data incomplete, missing data row")
            return 0.0
        except Exception as e:
            self.logger.error(f"Failed to process CSV data: {e}")
            return 0.0

    def get_top_k_results(self, k: int = 10) -> List[DataElement]:
        """
        Gets the top k results by result score

        Args:
            k: Number of results to return

        Returns:
            List[DataElement]: Top k elements sorted by result quality
        """
        with self.lock:
            try:
                # Get all elements
                cursor = self.collection.find({})
                elements_with_scores = []

                for doc in cursor:
                    # Remove MongoDB-specific fields
                    doc.pop('_id', None)
                    doc.pop('created_at', None)
                    doc.pop('updated_at', None)

                    element = self._safe_create_dataelement(doc)
                    # Calculate the score for the result
                    score = self._evaluate_result(element.result)
                    elements_with_scores.append((element, score))

                # Sort by score in descending order, then take the top k
                elements_with_scores.sort(key=lambda x: x[1], reverse=True)
                top_k_elements = [element for element, score in elements_with_scores[:k]]

                self.logger.info(f"Successfully got top-{k} results, found {len(top_k_elements)} results")
                return top_k_elements

            except PyMongoError as e:
                self.logger.error(f"Failed to get top-k results: {e}")
                return []
            except Exception as e:
                self.logger.error(f"Failed to get top-k results: {e}")
                return []

    def sample_from_range(self, a: int, b: int, k: int) -> List[DataElement]:
        """
        Randomly samples k results within the range [a, b] (after sorting)

        Args:
            a: Start position of the interval (counting from 1)
            b: End position of the interval (counting from 1, inclusive)
            k: Number of samples

        Returns:
            List[DataElement]: k randomly sampled elements
        """
        with self.lock:
            try:
                # Parameter validation
                if a < 1 or b < 1:
                    self.logger.error("Interval positions must start at 1")
                    return []

                if a > b:
                    self.logger.error("Start position can't be greater than the end position")
                    return []

                if k < 1:
                    self.logger.error("The number of samples must be greater than 0")
                    return []

                # Get all elements and sort
                cursor = self.collection.find({})
                elements_with_scores = []

                for doc in cursor:
                    # Remove MongoDB-specific fields
                    doc.pop('_id', None)
                    doc.pop('created_at', None)
                    doc.pop('updated_at', None)

                    element = self._safe_create_dataelement(doc)
                    # Calculate the score for the result
                    score = self._evaluate_result(element.result)
                    elements_with_scores.append((element, score))

                # Sort by score in descending order
                elements_with_scores.sort(key=lambda x: x[1], reverse=True)
                total_count = len(elements_with_scores)

                # Check if the interval is valid
                if a > total_count:
                    self.logger.warning(f"Start position {a} exceeds the total number of records {total_count}")
                    return []

                # Adjust the end position to not exceed the total number of records
                actual_b = min(b, total_count)

                # Extract elements within the interval (convert to 0-based index)
                range_elements = [element for element, score in elements_with_scores[a-1:actual_b]]
                range_size = len(range_elements)

                # If the number of elements in the interval is less than k, return all available elements
                if range_size <= k:
                    self.logger.info(f"There are only {range_size} elements in the interval [{a},{actual_b}], return all")
                    return range_elements

                # Randomly sample k elements
                import random
                sampled_elements = random.sample(range_elements, k)

                self.logger.info(f"Successfully sampled {k} results in the sorted interval [{a},{actual_b}]")
                return sampled_elements

            except PyMongoError as e:
                self.logger.error(f"Sampling from range failed: {e}")
                return []
            except Exception as e:
                self.logger.error(f"Sampling from range failed: {e}")
                return []

    def get_stats(self) -> Dict[str, Any]:
        """Gets the database statistics"""
        with self.lock:
            try:
                # Total number of records
                total_records = self.collection.count_documents({})

                # Number of unique names
                unique_names = len(self.collection.distinct("name"))

                # Database statistics
                db_stats = self.db.command("dbStats")
                collection_stats = self.db.command("collStats", self.collection_name)

                return {
                    "total_records": total_records,
                    "unique_names": unique_names,
                    "database_size": db_stats.get("dataSize", 0),
                    "collection_size": collection_stats.get("size", 0),
                    "index_size": collection_stats.get("totalIndexSize", 0),
                    "storage_size": collection_stats.get("storageSize", 0),
                    "average_object_size": collection_stats.get("avgObjSize", 0),
                    "database_name": self.database_name,
                    "collection_name": self.collection_name
                }

            except PyMongoError as e:
                self.logger.error(f"Failed to get statistics: {e}")
                return {}
            except Exception as e:
                self.logger.error(f"Failed to get statistics: {e}")
                return {}

    def repair_database(self) -> bool:
        """Repairs the database (cleans invalid data, rebuilds indexes)"""
        with self.lock:
            try:
                self.logger.info("Starting to repair the database...")

                # Find and delete invalid data
                invalid_count = 0

                # Delete documents missing required fields
                required_fields = ["name", "time", "program", "analysis", "result", "cognition", "log", "motivation", "index"]
                for field in required_fields:
                    query = {field: {"$exists": False}}
                    delete_result = self.collection.delete_many(query)
                    invalid_count += delete_result.deleted_count

                    # Delete documents with empty fields
                    query = {field: {"$in": [None, ""]}}
                    delete_result = self.collection.delete_many(query)
                    invalid_count += delete_result.deleted_count

                # Rebuild the indexes
                self.collection.drop_indexes()
                self._create_indexes()

                # Compact the collection (requires admin permissions)
                try:
                    self.db.command("compact", self.collection_name)
                    self.logger.info("Database compaction completed")
                except Exception as e:
                    self.logger.warning(f"Database compaction failed (may require admin permissions): {e}")

                self.logger.info(f"Database repair complete, deleted {invalid_count} invalid records")
                return True

            except PyMongoError as e:
                self.logger.error(f"Database repair failed: {e}")
                return False
            except Exception as e:
                self.logger.error(f"Database repair failed: {e}")
                return False

    def close(self):
        """Closes the database connection"""
        try:
            if hasattr(self, 'client'):
                self.client.close()
                self.logger.info("MongoDB connection closed")
        except Exception as e:
            self.logger.error(f"Failed to close connection: {e}")

    def __enter__(self):
        """Context manager enter method"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method"""
        self.close()

    def search_similar_motivations(self, query_motivation: str, k: int = 5) -> List[tuple]:
        """
        Searches for the k most similar elements given a motivation (using FAISS acceleration)

        Args:
            query_motivation: The motivation text to query with.
            k: The number of results to return.

        Returns:
            List[tuple]: A list of (DataElement, similarity_score) tuples, sorted in descending order
            of similarity.
        """
        with self.lock:
            try:
                # Calculate the embedding of the query text
                embedding_service = get_embedding_service()
                query_embedding = embedding_service.get_single_embedding(query_motivation)

                if not query_embedding:
                    self.logger.error("Failed to calculate the embedding for the query text")
                    return []

                # Use FAISS for fast search
                faiss_manager = get_faiss_manager()
                faiss_results = faiss_manager.search_similar(query_embedding, k)

                if not faiss_results:
                    self.logger.info("FAISS search found no results")
                    return []

                # Get the full data from MongoDB based on the FAISS results
                results = []
                for doc_id, similarity_score in faiss_results:
                    try:
                        # Convert the string ID to an ObjectId
                        from bson import ObjectId
                        object_id = ObjectId(doc_id)

                        # Get the document from MongoDB
                        doc = self.collection.find_one({"_id": object_id})
                        if doc:
                            # Remove MongoDB-specific fields
                            doc.pop('_id', None)
                            doc.pop('created_at', None)
                            doc.pop('updated_at', None)

                            element = self._safe_create_dataelement(doc)
                            results.append((element, similarity_score))
                        else:
                            self.logger.warning(f"Document ID not found: {doc_id}")

                    except Exception as e:
                        self.logger.warning(f"Failed to process search result {doc_id}: {e}")
                        continue

                self.logger.info(f"FAISS search completed, found {len(results)} valid results")
                return results

            except Exception as e:
                self.logger.error(f"Searching similar motivations failed: {e}")
                return []

    def rebuild_faiss_index(self) -> bool:
        """
        Rebuilds the FAISS index from MongoDB data.
        Used for initialization or repairing the FAISS index.

        Returns:
            bool: Whether the rebuild was successful.
        """
        with self.lock:
            try:
                self.logger.info("Starting to rebuild the FAISS index...")

                # Get all data that has an embedding
                cursor = self.collection.find({
                    "motivation_embedding": {"$exists": True, "$ne": None}
                })

                vectors_data = []
                for doc in cursor:
                    doc_id = str(doc['_id'])
                    embedding = doc.get('motivation_embedding')
                    if embedding and len(embedding) == 1024:  # Ensure the vector dimensions are correct. Updated to 1024 for self hosted model
                        vectors_data.append((embedding, doc_id))

                # Rebuild the FAISS index
                faiss_manager = get_faiss_manager()
                faiss_manager.rebuild_index(vectors_data)

                self.logger.info(f"FAISS index rebuild completed, containing {len(vectors_data)} vectors")
                return True

            except Exception as e:
                self.logger.error(f"Failed to rebuild FAISS index: {e}")
                return False

    def get_faiss_stats(self) -> Dict[str, Any]:
        """Gets the FAISS index statistics"""
        try:
            faiss_manager = get_faiss_manager()
            return faiss_manager.get_stats()
        except Exception as e:
            self.logger.error(f"Failed to get FAISS statistics: {e}")
            return {}

    def clean_faiss_orphans(self) -> Dict[str, Any]:
        """Cleans orphaned vectors from FAISS"""
        try:
            faiss_manager = get_faiss_manager()
            result = faiss_manager.clean_orphan_vectors()
            self.logger.info(f"FAISS orphan vector cleanup completed: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to clean FAISS orphan vectors: {e}")
            return {"error": str(e)}

    def _get_recent_elements(self, limit: int = 50) -> List[DataElement]:
        """
        Gets recently added data elements

        Args:
            limit: The number of elements to retrieve.

        Returns:
            List[DataElement]: A list of recent data elements.
        """
        try:
            # Sort by created_at in descending order and get the latest 'limit' records
            cursor = self.collection.find({}).sort("created_at", -1).limit(limit)
            elements = []

            for doc in cursor:
                # Remove MongoDB-specific fields
                doc.pop('_id', None)
                doc.pop('created_at', None)
                doc.pop('updated_at', None)

                element = self._safe_create_dataelement(doc)
                elements.append(element)

            return elements

        except Exception as e:
            self.logger.error(f"Failed to get recent data: {e}")
            return []

    def get_elements_with_score(self) -> List[DataElement]:
        """
        Gets all elements that have valid scores, and sorts them by score in descending order.

        Returns:
            List[DataElement]: A list of elements sorted by score.
        """
        with self.lock:
            try:
                # Query for all documents where the score field exists and is not null
                cursor = self.collection.find(
                    {"score": {"$exists": True, "$ne": None}}
                ).sort("score", -1)

                elements = []
                for doc in cursor:
                    doc.pop('_id', None)
                    doc.pop('created_at', None)
                    doc.pop('updated_at', None)
                    
                    element = self._safe_create_dataelement(doc)
                    elements.append(element)

                self.logger.info(f"Successfully retrieved {len(elements)} scored elements")
                return elements

            except PyMongoError as e:
                self.logger.error(f"Failed to get scored elements: {e}")
                return []

    async def get_or_calculate_element_score(self, index: int) -> Optional[float]:
        """
        Gets the element's score, and if not present, computes and caches it.

        Args:
            index: Element index.

        Returns:
            Optional[float]: Element's score.
        """
        element = self.get_by_index(index)
        if not element:
            return None

        if element.score is not None:
            self.logger.info(f"Found cached score for element {index} in the database: {element.score}")
            return element.score

        self.logger.info(f"No cached score found for element {index}, calculating...")
        candidate_manager = get_candidate_manager()
        # _evaluate_element will save the score via the callback.
        score = await candidate_manager._evaluate_element(element)
        return score

    async def rebuild_candidates_from_scored_elements(self) -> Dict[str, Any]:
        """
        Rebuilds the candidate set using all scored elements in the database.

        Returns:
            Dict[str, Any]: Update result.
        """
        try:
            self.logger.info("Starting to rebuild the candidate set from scored elements...")

            scored_elements = self.get_elements_with_score()
            if not scored_elements:
                self.logger.info("No scored elements found in the database; no update performed.")
                return {"message": "No scored elements found in DB. Candidates not updated."}

            top_elements = scored_elements[:50]

            candidate_manager = get_candidate_manager()
            result = candidate_manager.replace_candidates(top_elements)

            self.logger.info("Successfully rebuilt the candidate set from scored elements.")
            return result

        except Exception as e:
            self.logger.error(f"Failed to rebuild the candidate set from scored elements: {e}")
            return {"error": str(e)}

    # Candidate set related methods
    def get_candidate_top_k(self, k: int = 10) -> List[DataElement]:
        """Gets the top-k elements from the candidate set"""
        try:
            candidate_manager = get_candidate_manager()
            return candidate_manager.get_top_k(k)
        except Exception as e:
            self.logger.error(f"Failed to get candidate top-k: {e}")
            return []

    def get_all_candidates_with_scores(self) -> List[DataElement]:
        """Gets all candidate set elements and their scores"""
        try:
            candidate_manager = get_candidate_manager()
            return candidate_manager.get_all_candidates()
        except Exception as e:
            self.logger.error(f"Failed to get all candidates: {e}")
            return []

    def candidate_sample_from_range(self, a: int, b: int, k: int) -> List[DataElement]:
        """Samples within the candidate set in a specified range (returns DataElements)."""
        try:
            candidate_manager = get_candidate_manager()
            raw_response = candidate_manager.sample_from_range(a, b, k)
            return [self._safe_create_dataelement(item) for item in raw_response]
        except Exception as e:
            self.logger.error(f"Candidate set sampling from range failed: {e}")
            return []
        
    async def add_to_candidates(self, element: DataElement) -> bool:
        """Manually adds an element to the candidate set"""
        try:
            candidate_manager = get_candidate_manager()
            return await candidate_manager.add_element(element)
        except Exception as e:
            self.logger.error(f"Failed to add to candidate set: {e}")
            return False

    def delete_candidate_by_index(self, index: int) -> bool:
        """Deletes a candidate by index"""
        try:
            candidate_manager = get_candidate_manager()
            return candidate_manager.delete_by_index(index)
        except Exception as e:
            self.logger.error(f"Failed to delete from candidate set: {e}")
            return False

    def delete_candidate_by_name(self, name: str) -> int:
        """Deletes candidates by name"""
        try:
            candidate_manager = get_candidate_manager()
            return candidate_manager.delete_by_name(name)
        except Exception as e:
            self.logger.error(f"Failed to delete from candidate set: {e}")
            return 0

    async def update_candidate(self, element: DataElement) -> bool:
        """Updates a candidate element"""
        try:
            candidate_manager = get_candidate_manager()
            return await candidate_manager.update_element(element)
        except Exception as e:
            self.logger.error(f"Failed to update candidate element: {e}")
            return False

    async def force_update_candidates(self) -> Dict[str, Any]:
        """Forces an update of the candidate set"""
        try:
            candidate_manager = get_candidate_manager()
            recent_elements = self._get_recent_elements(300)  # Get more data for updating
            return await candidate_manager.update_candidates(recent_elements)
        except Exception as e:
            self.logger.error(f"Failed to force update candidate set: {e}")
            return {"error": str(e)}

    def get_candidate_stats(self) -> Dict[str, Any]:
        """Gets candidate set statistics"""
        try:
            candidate_manager = get_candidate_manager()
            return candidate_manager.get_stats()
        except Exception as e:
            self.logger.error(f"Failed to get candidate set statistics: {e}")
            return {"error": str(e)}

    def get_candidate_new_data_count(self) -> int:
        """Gets the candidate set's new data count"""
        try:
            candidate_manager = get_candidate_manager()
            return candidate_manager.get_new_data_count()
        except Exception as e:
            self.logger.error(f"Failed to get candidate set new data count: {e}")
            return -1

    def clear_candidates(self) -> bool:
        """Clears the candidate set"""
        try:
            candidate_manager = get_candidate_manager()
            return candidate_manager.clear()
        except Exception as e:
            self.logger.error(f"Failed to clear candidate set: {e}")
            return False

    def uct_select_node(self, c_param: float = 1.414) -> Optional[DataElement]:
        """
        Selects a node using the UCT algorithm
        UCT formula: selection score = score + C * sqrt(ln(N_total) / N_node)

        Args:
            c_param: Exploration parameter, default is sqrt(2) â‰ˆ 1.414.

        Returns:
            DataElement: The selected node, or None if no nodes are available.
        """
        with self.lock:
            try:
                import math

                # Get all nodes
                cursor = self.collection.find({})
                all_elements = []
                for doc in cursor:
                    # Remove MongoDB-specific fields
                    doc.pop('_id', None)
                    doc.pop('created_at', None)
                    doc.pop('updated_at', None)
                    
                    element = self._safe_create_dataelement(doc)
                    all_elements.append(element)

                if not all_elements:
                    self.logger.warning("No nodes found")
                    return None

                # Calculate total number of nodes
                n_total = len(all_elements)

                if n_total <= 1:
                    # If there is only one or no nodes, just return the first node
                    return all_elements[0] if all_elements else None

                # N_total = total nodes - 1 (as total exploration times)
                total_explorations = n_total - 1

                best_element = None
                best_uct_score = float('-inf')

                # Calculate the UCT score for all elements
                for element in all_elements:
                    # Calculate the base score of the node (using _evaluate_result)
                    base_score = self._evaluate_result(element.result)

                    # Calculate the number of child nodes for the current node
                    children = self.get_children(element.index)
                    n_node = len(children)

                    # If the node was never explored (no children), give the highest priority
                    if n_node == 0:
                        uct_score = float('inf')  # Not explored nodes have infinitely high UCT score
                    else:
                        # Calculate the UCT score
                        exploration_term = c_param * math.sqrt(math.log(total_explorations) / n_node)
                        uct_score = base_score + exploration_term

                    self.logger.debug(f"Node {element.index} ({element.name}): "
                                    f"base_score={base_score:.4f}, "
                                    f"n_node={n_node}, "
                                    f"uct_score={uct_score:.4f}")

                    # Update the best node
                    if uct_score > best_uct_score:
                        best_uct_score = uct_score
                        best_element = element

                if best_element:
                    self.logger.info(f"UCT algorithm selected node: {best_element.name} (index: {best_element.index}), "
                                   f"UCT score: {best_uct_score:.4f}")
                else:
                    self.logger.warning("UCT algorithm failed to select a node")

                return best_element

            except Exception as e:
                self.logger.error(f"UCT node selection failed: {e}")
                return None

    def get_uct_scores(self, c_param: float = 1.414) -> List[Dict[str, Any]]:
        """
        Gets UCT score details for all nodes

        Args:
            c_param: Exploration parameter

        Returns:
            List[Dict]: List containing UCT score details for each node
        """
        with self.lock:
            try:
                import math

                # Get all nodes
                cursor = self.collection.find({})
                all_elements = []
                for doc in cursor:
                    # Remove MongoDB-specific fields
                    doc.pop('_id', None)
                    doc.pop('created_at', None)
                    doc.pop('updated_at', None)
                    
                    element = self._safe_create_dataelement(doc)
                    all_elements.append(element)

                if not all_elements:
                    return []

                n_total = len(all_elements)
                total_explorations = max(n_total - 1, 1)  # Avoid division by zero

                uct_scores = []
                for element in all_elements:
                    # Calculate base score
                    base_score = self._evaluate_result(element.result)

                    # Calculate the number of child nodes
                    children = self.get_children(element.index)
                    n_node = len(children)

                    # Calculate UCT score
                    if n_node == 0:
                        uct_score = float('inf')
                        exploration_term = float('inf')
                    else:
                        exploration_term = c_param * math.sqrt(math.log(total_explorations) / n_node)
                        uct_score = base_score + exploration_term

                    uct_scores.append({
                        "index": element.index,
                        "name": element.name,
                        "base_score": base_score,
                        "n_node": n_node,
                        "exploration_term": exploration_term if exploration_term != float('inf') else "infinite",
                        "uct_score": uct_score if uct_score != float('inf') else "infinite",
                        "summary": element.summary
                    })

                # Sort by UCT score in descending order
                uct_scores.sort(key=lambda x: x["uct_score"] if x["uct_score"] != "infinite" else float('inf'),
                               reverse=True)

                return uct_scores

            except Exception as e:
                self.logger.error(f"Failed to get UCT scores: {e}")
                return []

    def get_contextual_nodes(self, parent_index: int) -> Dict[str, Any]:
        """
        Gets contextual nodes based on the parent index (parent, grandparent, strongest siblings)

        Args:
            parent_index: Index of the direct parent of the new node

        Returns:
            Dict: A dictionary containing the contextual node information
        """
        with self.lock:
            context = {
                "direct_parent": None,
                "strongest_siblings": [],
                "grandparent": None
            }

            # 1. Get the direct parent node
            direct_parent = self.get_by_index(parent_index)
            if not direct_parent:
                self.logger.warning(f"Failed to get contextual nodes: parent node {parent_index} does not exist")
                return context

            context["direct_parent"] = direct_parent

            # 2. Get the grandparent node
            if direct_parent.parent is not None:
                grandparent = self.get_by_index(direct_parent.parent)
                context["grandparent"] = grandparent

            # 3. Get the strongest siblings (new node's siblings)
            siblings = self.get_children(parent_index)
            if siblings:
                siblings_with_visit_count = []
                for sibling in siblings:
                    # Use the number of children as a proxy for visit count
                    visit_count = len(self.get_children(sibling.index))
                    siblings_with_visit_count.append((sibling, visit_count))

                # Sort in descending order by visit count
                siblings_with_visit_count.sort(key=lambda x: x[1], reverse=True)

                # Extract the strongest two siblings
                strongest = [sibling for sibling, count in siblings_with_visit_count[:2]]
                context["strongest_siblings"] = strongest

            return context

    def set_parent(self, child_index: int, parent_index: Optional[int]) -> bool:
        """
        Sets the parent node for a specified element

        Args:
            child_index: The index of the child node.
            parent_index: The index of the parent node, or None to set as a root node.

        Returns:
            bool: Whether the set was successful.
        """
        with self.lock:
            try:
                # Validate that the child node exists
                child_element = self.get_by_index(child_index)
                if child_element is None:
                    self.logger.error(f"Child node index={child_index} does not exist")
                    return False

                # Validate that the parent node exists (if specified)
                if parent_index is not None:
                    parent_element = self.get_by_index(parent_index)
                    if parent_element is None:
                        self.logger.error(f"Parent node index={parent_index} does not exist")
                        return False

                # Check for circular references
                if self._would_create_cycle(child_index, parent_index):
                    self.logger.error(f"Setting the parent node would create a circular reference: child={child_index}, parent={parent_index}")
                    return False

                # Update the parent field in the database
                result = self.collection.update_one(
                    {"index": child_index},
                    {"$set": {"parent": parent_index, "updated_at": datetime.now()}}
                )

                if result.modified_count > 0:
                    self.logger.info(f"Successfully set the parent node of element {child_index} to {parent_index}")
                    return True
                else:
                    self.logger.error(f"Failed to set the parent node, element with index={child_index} not found")
                    return False

            except PyMongoError as e:
                self.logger.error(f"MongoDB error occurred while setting the parent node: {e}")
                return False
            except Exception as e:
                self.logger.error(f"An unknown error occurred while setting the parent node: {e}")
                return False

    def _would_create_cycle(self, child_index: int, parent_index: int) -> bool:
        """
        Checks if setting the parent-child relationship would create a circular reference

        Args:
            child_index: The index of the child node.
            parent_index: The index of the parent node.

        Returns:
            bool: True if a cycle would be created, False otherwise.
        """
        try:
            # Start traversing upwards from parent_index to check if child_index is encountered
            current_index = parent_index
            visited = set()

            while current_index is not None:
                if current_index == child_index:
                    return True  # Cycle detected

                if current_index in visited:
                    # Detected another type of cycle, but not involving child_index
                    break

                visited.add(current_index)

                # Get the parent of the current node
                current_element = self.get_by_index(current_index)
                if current_element is None:
                    break

                current_index = current_element.parent

            return False

        except Exception as e:
            self.logger.warning(f"Error encountered when checking for circular references: {e}")
            return True  # Return True for safety in case of an error

    def get_children(self, parent_index: int) -> List[DataElement]:
        """
        Gets all child nodes of a specified node

        Args:
            parent_index: The index of the parent node

        Returns:
            List[DataElement]: A list of child nodes
        """
        with self.lock:
            try:
                # Query all documents with the specified parent index
                cursor = self.collection.find({"parent": parent_index}).sort("index", ASCENDING)
                children = []

                for doc in cursor:
                    # Remove MongoDB-specific fields
                    doc.pop('_id', None)
                    doc.pop('created_at', None)
                    doc.pop('updated_at', None)
                    
                    element = self._safe_create_dataelement(doc)
                    children.append(element)

                return children

            except PyMongoError as e:
                self.logger.error(f"Failed to get children: {e}")
                return []
            except Exception as e:
                self.logger.error(f"Failed to get children: {e}")
                return []

    def get_root_nodes(self) -> List[DataElement]:
        """
        Gets all root nodes (nodes with parent set to None)

        Returns:
            List[DataElement]: A list of root nodes
        """
        with self.lock:
            try:
                # Query all documents with parent set to None
                cursor = self.collection.find({"parent": None}).sort("index", ASCENDING)
                roots = []

                for doc in cursor:
                    # Remove MongoDB-specific fields
                    doc.pop('_id', None)
                    doc.pop('created_at', None)
                    doc.pop('updated_at', None)
                    
                    element = self._safe_create_dataelement(doc)
                    roots.append(element)

                return roots

            except PyMongoError as e:
                self.logger.error(f"Failed to get root nodes: {e}")
                return []
            except Exception as e:
                self.logger.error(f"Failed to get root nodes: {e}")
                return []

    def get_tree_path(self, index: int) -> List[DataElement]:
        """
        Gets the path from the root node to a specified node

        Args:
            index: The index of the target node

        Returns:
            List[DataElement]: The path from the root, or an empty list if the node doesn't exist
        """
        with self.lock:
            try:
                path = []
                current_index = index
                visited = set()

                # Traverse upwards to the root node
                while current_index is not None:
                    if current_index in visited:
                        self.logger.warning(f"Cycle detected, stopping traversal: {current_index}")
                        break

                    visited.add(current_index)

                    current_element = self.get_by_index(current_index)
                    if current_element is None:
                        break

                    path.insert(0, current_element)  # Insert at the beginning
                    current_index = current_element.parent

                return path

            except Exception as e:
                self.logger.error(f"Failed to get tree path: {e}")
                return []

    def get_tree_structure(self, root_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Gets the complete information of the tree structure.

        Args:
            root_index: The index of the root node. If None, retrieves all trees.

        Returns:
            Dict: Tree structure information.
        """
        with self.lock:
            try:
                def build_tree_node(element: DataElement) -> Dict[str, Any]:
                    """Recursively builds a tree node"""
                    children = self.get_children(element.index)
                    return {
                        "element": {
                            "index": element.index,
                            "name": element.name,
                            "time": element.time,
                            "parent": element.parent
                        },
                        "children": [build_tree_node(child) for child in children]
                    }

                if root_index is not None:
                    # Get the tree structure for the specified root node
                    root_element = self.get_by_index(root_index)
                    if root_element is None:
                        return {"error": f"Root node index={root_index} does not exist"}

                    return build_tree_node(root_element)
                else:
                    # Get the tree structure for all root nodes
                    root_nodes = self.get_root_nodes()
                    return {
                        "trees": [build_tree_node(root) for root in root_nodes],
                        "total_roots": len(root_nodes)
                    }

            except Exception as e:
                self.logger.error(f"Failed to get tree structure: {e}")
                return {"error": str(e)}

    def update_element_score(self, index: int, score: float) -> bool:
        """Updates the score of an element in the database."""
        with self.lock:
            try:
                result = self.collection.update_one(
                    {"index": index},
                    {"$set": {"score": score, "updated_at": datetime.now()}}
                )

                if result.modified_count > 0:
                    self.logger.info(f"Successfully updated the score for element {index} to {score:.4f}")
                    return True
                elif result.matched_count == 0:
                    self.logger.warning(f"Failed to update score: could not find element with index={index}")
                    return False

                return True  # score didn't change already exists

            except PyMongoError as e:
                self.logger.error(f"Failed to update the score for element {index}: {e}")
                return False

    def clean_invalid_result_elements(self) -> Dict[str, Any]:
        """Cleans invalid elements with incomplete or only header in result field."""
        
        def _is_csv_row_complete(csv_string: str) -> bool:
            """
            Checks if the data row of a CSV string (the second row) is complete, i.e., exists and contains no empty values.
            """
            if not isinstance(csv_string, str) or not csv_string.strip():
                return False

            with io.StringIO(csv_string) as f:
                try:
                    reader = csv.reader(f)
                    _ = next(reader)  # Skip header
                    data_row = next(reader)  # Read data row

                    if not data_row:
                        return False  # Row is empty

                    # Check for any empty strings in the data row's values
                    # The first value can be a name, so we check all of them.
                    for value in data_row:
                        if not value.strip():
                            return False  # Found an empty value

                    return True  # Row is complete and has no empty values

                except StopIteration:
                    # No data row at all
                    return False
                except Exception:
                    # On other errors, assume valid to be safe
                    return True

        with self.lock:
            self.logger.info("Starting to scan and clean invalid data (strict mode)...")
            count_before = self.collection.count_documents({})

            all_elements_cursor = self.collection.find({}, {"index": 1, "result": 1, "parent": 1})
            invalid_elements_map = {}

            for doc in all_elements_cursor:
                result_field = doc.get("result", {})
                train_csv = result_field.get("train", "")
                test_csv = result_field.get("test", "")

                if not _is_csv_row_complete(train_csv) or not _is_csv_row_complete(test_csv):
                    self.logger.debug(f"Marked as invalid element Index {doc['index']}: train_valid={_is_csv_row_complete(train_csv)}, test_valid={_is_csv_row_complete(test_csv)}")
                    invalid_elements_map[doc["index"]] = doc.get("parent")

            if not invalid_elements_map:
                self.logger.info("Scan complete, no invalid elements found.")
                return {"cleaned_count": 0, "reparented_children": 0, "count_before": count_before, "count_after": count_before, "message": "No invalid elements found."}

            self.logger.info(f"Found {len(invalid_elements_map)} invalid elements to clean.")
            reparented_children_count = 0

            # Reparent all children nodes
            for index, parent_of_deleted in invalid_elements_map.items():
                if not isinstance(index, int):
                    continue

                update_result = self.collection.update_many(
                    {"parent": index},
                    {"$set": {"parent": parent_of_deleted, "updated_at": datetime.now()}}
                )

                reparented_children_count += update_result.modified_count

            # Delete all invalid elements at once
            invalid_indices = list(invalid_elements_map.keys())
            delete_result = self.collection.delete_many({"index": {"$in": invalid_indices}})
            cleaned_count = delete_result.deleted_count

            # Synchronize candidate sets and FAISS
            candidate_manager = get_candidate_manager()
            for index in invalid_indices:
                candidate_manager.delete_by_index(index)

            # Note: FAISS is not synchronized here because getting _id requires an extra query.
            # FAISS can be repaired by rebuilding it later.

            count_after = self.collection.count_documents({})

            self.logger.info(f"Cleaning completed. Deleted {cleaned_count} invalid elements and reparented {reparented_children_count} children.")

            return {
                "count_before": count_before,
                "cleaned_count": cleaned_count,
                "reparented_children": reparented_children_count,
                "count_after": count_after
            }


# Factory function to create database instances with different configurations
def create_mongo_database(
    host: str = "localhost",
    port: int = 27017,
    database_name: str = "myapp",
    collection_name: str = "data_elements",
    username: str = None,
    password: str = None
) -> MongoDatabase:
    """
    Factory function to create MongoDB database instances

    Args:
        host: MongoDB host address
        port: MongoDB port
        database_name: Database name.
        collection_name: Collection name.
        username: Username.
        password: Password.

    Returns:
        MongoDatabase instance
    """
    # Use environment variable if available, otherwise fall back to constructed string
    connection_string = os.environ.get('MONGODB_URI')
    if not connection_string:
        connection_string = f"mongodb://{host}:{port}"
    
    return MongoDatabase(
        connection_string=connection_string,
        database_name=database_name,
        collection_name=collection_name,
        username=username,
        password=password
    )
