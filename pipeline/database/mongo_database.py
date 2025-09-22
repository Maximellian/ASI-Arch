#!/usr/bin/env python3
"""
MongoDB API client wrapper class.
Provides complete MongoDB database API functionality.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from pipeline.config import Config
from pipeline.database.element import DataElement

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ApiResponse:
    """API response model."""
    success: bool
    message: str
    data: Optional[Any] = None


@dataclass
class StatsInfo:
    """Database statistics information model."""
    total_records: int
    unique_names: int
    database_size: int
    collection_size: int
    index_size: int
    storage_size: int
    average_object_size: int
    database_name: str
    collection_name: str


class MongoDBAPIException(Exception):
    """MongoDB API exception class."""
    
    def __init__(self, message: str, status_code: int = None, detail: str = None):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)


class MongoDBAPIClient:
    """MongoDB API client wrapper class."""
    
    def __init__(self, 
                 base_url: str = Config.DATABASE,
                 timeout: int = 30,
                 verify_ssl: bool = True):
        """
        Initialize API client.
        
        Args:
            base_url: API service base URL
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificate
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.headers = {"Content-Type": "application/json"}
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test API connection."""
        try:
            response = self._make_request("GET", "/health")
            if response.get("status") == "healthy":
                logger.info("✅ API connection test successful")
                return True
            else:
                logger.warning("⚠️ API connection abnormal")
                return False
        except Exception as e:
            logger.error(f"❌ API connection failed: {e}")
            raise MongoDBAPIException(f"Unable to connect to API service: {e}")
    
    def _make_request(self, 
                     method: str, 
                     endpoint: str, 
                     params: Dict = None, 
                     data: Dict = None) -> Dict[str, Any]:
        """
        Make HTTP request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: URL parameters
            data: Request body data
            
        Returns:
            Response data
            
        Raises:
            MongoDBAPIException: API request exception
        """
        url = f"{self.base_url}{endpoint}"
        
        # Prepare request parameters
        kwargs = {
            "timeout": self.timeout,
            "verify": self.verify_ssl,
            "params": params
        }
        
        if data is not None:
            kwargs["headers"] = self.headers
            kwargs["data"] = json.dumps(data)
        
        try:
            response = requests.request(method, url, **kwargs)
            
            # Check response status
            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    detail = error_data.get("detail", response.text)
                except:
                    detail = response.text
                
                raise MongoDBAPIException(
                    f"API request failed: {response.status_code}",
                    status_code=response.status_code,
                    detail=detail
                )
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise MongoDBAPIException(f"Network request failed: {e}")
        except json.JSONDecodeError as e:
            raise MongoDBAPIException(f"Response parsing failed: {e}")
    
    # ========== Basic Interface ==========
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get API basic information."""
        return self._make_request("GET", "/")
    
    def health_check(self) -> Dict[str, Any]:
        """Health check."""
        return self._make_request("GET", "/health")
    
    # ========== Data Operation Interface ==========
    
    def add_element(self,
                   time: str,
                   name: str,
                   result: Dict[str, str],
                   program: str,
                   motivation: str,
                   analysis: str,
                   cognition: str,
                   log: str,
                   parent: int = None,
                   summary: str = None) -> ApiResponse:
        """
        Add data element.
        
        Args:
            time: Timestamp
            name: Name
            result: Result
            program: Program
            motivation: Motivation
            analysis: Analysis
            cognition: Cognition
            log: Log
            parent: Parent node index
            summary: Summary
            
        Returns:
            API response object
        """
        data = {
            "time": time,
            "name": name,
            "result": result,
            "program": program,
            "motivation": motivation,
            "analysis": analysis,
            "cognition": cognition,
            "log": log,
        }
        
        if parent is not None:
            data["parent"] = int(parent)
        if summary is not None:
            data["summary"] = summary
        
        response = self._make_request("POST", "/elements", data=data)
        return ApiResponse(
            success=response.get("success", False),
            message=response.get("message", ""),
            data=response.get("data")
        )
    
    def add_element_from_dict(self, element_data: Dict[str, str]) -> ApiResponse:
        """Add data element from dictionary."""
        return self.add_element(
            time=element_data["time"],
            name=element_data["name"],
            result=element_data["result"],
            program=element_data["program"],
            motivation=element_data["motivation"],
            analysis=element_data["analysis"],
            cognition=element_data["cognition"],
            log=element_data["log"],
            parent=element_data.get("parent"),
            summary=element_data.get("summary")
        )
    
    def sample_element(self) -> Optional[DataElement]:
        """
        Randomly sample a data element.
            
        Returns:
            Data element object or None
        """
        try:
            response = self._make_request("GET", "/elements/sample")
            return DataElement(**response)
        except MongoDBAPIException as e:
            if e.status_code == 404:
                return None
            raise
        
    def get_elements_by_name(self, name: str) -> List[DataElement]:
        """
        Get data element list by name.
        
        Args:
            name: Element name
            
        Returns:
            Data element list
        """
        response = self._make_request("GET", f"/elements/by-name/{name}")
        return [DataElement(**item) for item in response]
    
    def get_elements_by_index(self, index: int) -> Optional[DataElement]:
        """
        Get data element by index.
        
        Args:
            index: Element index
            
        Returns:
            Data element object or None
        """
        try:
            response = self._make_request("GET", f"/elements/by-index/{index}")
            return DataElement(**response)
        except MongoDBAPIException as e:
            if e.status_code == 404:
                return None
            raise
    
    def delete_elements_by_index(self, index: int) -> ApiResponse:
        """
        Delete data element by index.

        Args:
            index: Element index

        Returns:
            API response object indicating whether the operation was successful
        """
        try:
            response = self._make_request("DELETE", f"/elements/by-index/{index}")
            return ApiResponse(
                success=response.get("success", False),
                message=response.get("message", ""),
                data=response.get("data")
            )
        except MongoDBAPIException as e:
            if e.status_code == 404:
                return ApiResponse(
                    success=False, 
                    message=e.detail or f"Element with index={index} not found"
                )
            raise e
        
    def delete_elements_by_name(self, name: str) -> ApiResponse:
        """Delete data element by name."""
        try:
            response = self._make_request("DELETE", f"/elements/by-name/{name}")
            return ApiResponse(
                success=response.get("success", False),
                message=response.get("message", ""),
                data=response.get("data")
            )
        except MongoDBAPIException as e:
            if e.status_code == 404:
                return ApiResponse(
                    success=False, 
                    message=e.detail or f"Element with name={name} not found"
                )
            raise e
        
    def search_similar_motivations(self, motivation: str, top_k: int = 5) -> List[DataElement]:
        """
        Search for most similar data elements based on motivation text.
        
        Args:
            motivation: Motivation text to search for
            top_k: Number of similar results to return, must be between 1-20
            
        Returns:
            List of similar data elements
            
        Raises:
            ValueError: When top_k parameter is not within valid range
        """
        if top_k <= 0 or top_k > 20:
            raise ValueError("top_k must be between 1-20")
        
        params = {"motivation": motivation, "top_k": top_k}
        response = self._make_request("GET", "/elements/search-similar", params=params)
        return [DataElement.from_dict(item) for item in response]
    
    def get_top_k_results(self, k: int) -> List[DataElement]:
        """
        Get top k results with best performance.
        
        Args:
            k: Number of results to return
            
        Returns:
            List of data elements
        """
        if k <= 0 or k > 1000:
            raise ValueError("k must be between 1-1000")
        
        response = self._make_request("GET", f"/elements/top-k/{k}")
        return [DataElement(**item) for item in response]
    
    def sample_from_range(self, a: int, b: int, k: int) -> List[DataElement]:
        """
        Randomly sample k results from the a-th to b-th results in sorted order.
        
        Args:
            a: Start position
            b: End position
            k: Number of samples
            
        Returns:
            List of data elements
        """
        if a <= 0 or b <= 0 or k <= 0:
            raise ValueError("All parameters must be positive integers")
        if a > b:
            raise ValueError("Start position cannot be greater than end position")
        if k > 1000:
            raise ValueError("Sample count cannot exceed 1000")
        if (b - a + 1) > 10000:
            raise ValueError("Range cannot exceed 10000")
        
        response = self._make_request("GET", f"/elements/sample-range/{a}/{b}/{k}")
        return [DataElement(**item) for item in response]
    
    def uct_select_node(self, c_param: float = 1.414) -> Optional[DataElement]:
        """
        Select a node using UCT algorithm.
        
        Args:
            c_param: Exploration parameter for UCT algorithm, must be positive and not exceed 10
            
        Returns:
            Selected data element object or None if no selectable nodes
            
        Raises:
            ValueError: When c_param parameter is not within valid range
        """
        if c_param <= 0:
            raise ValueError("c_param must be positive")
        if c_param > 10:
            raise ValueError("c_param cannot exceed 10")
        
        params = {"c_param": c_param}
        try:
            response = self._make_request("GET", "/elements/uct-select", params=params)
            return DataElement(**response)
        except MongoDBAPIException as e:
            if e.status_code == 404:
                return None
            raise
    
    def get_stats(self) -> StatsInfo:
        """
        Get database statistics.
            
        Returns:
            Statistics information object
        """
        response = self._make_request("GET", "/stats")
        return StatsInfo(**response)
    
    def repair_database(self) -> ApiResponse:
        """
        Repair database.
            
        Returns:
            API response object
        """
        response = self._make_request("POST", "/repair")
        return ApiResponse(
            success=response.get("success", False),
            message=response.get("message", ""),
            data=response.get("data")
        )

    def get_analyse_elements(self, parent_index: int) -> dict:
        """
        Get context nodes (parent, grandparent, strongest siblings) based on parent index.

        Args:
            parent_index: Parent node index

        Returns:
            Dictionary of context node elements
        """
        try:
            response = self._make_request("GET", f"/elements/context/{parent_index}")

            if not response.get("success"):
                logger.warning(
                    f"API call to get context for {parent_index} was not successful: "
                    f"{response.get('message')}"
                )
                return {}

            context_data = response.get("data")
            if not context_data:
                return {}

            return context_data
        except MongoDBAPIException as e:
            if e.status_code == 404:
                return {}
            raise

    def candidate_sample_from_range(self, a: int, b: int, k: int) -> List[DataElement]:
        """
        Randomly sample k elements from specified range in candidate set.

        Args:
            a: Start position
            b: End position
            k: Number of samples

        Returns:
            List of data elements

        Raises:
            ValueError: When parameters are not within valid range
        """
        if a <= 0 or b <= 0 or k <= 0:
            raise ValueError("All parameters must be positive integers")
        if a > b:
            raise ValueError("Start position cannot be greater than end position")
        if k > 50:
            raise ValueError("Sample count cannot exceed 50")
        if (b - a + 1) > 50:
            raise ValueError("Range cannot exceed 50")

        response = self._make_request("GET", f"/candidates/sample-range/{a}/{b}/{k}")
        return [DataElement(**item) for item in response]


# ========== Convenience Factory Function ==========

def create_client(base_url: str = Config.DATABASE) -> MongoDBAPIClient:
    """Create API client."""
    return MongoDBAPIClient(base_url=base_url)
