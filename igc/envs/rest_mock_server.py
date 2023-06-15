"""
This class represents a mock server that env sends.

MockServer class represents a mock server that simulates
the behavior of an environment that sends requests.

It is designed to handle various HTTP methods such as GET, POST, PUT, DELETE, PATCH, and HEAD.

For all GET requests, we always respond with a 200 status code.
For actions that mutate a state, we check the callback, to synthetically
mutate a state and inform the agent whether
it's okay or not.

Here's a description of what it does:

The server always responds with a status code of 200 for GET requests.
For actions that mutate a state, the server checks a callback function to artificially
mutate the state and inform the agent whether the mutation is valid or not.

The class provides methods to register callbacks for specific endpoints
and methods, allowing customization of the server's behavior.
It supports generating generic error responses with status code 400.
The server can be configured to simulate an HTTP 500 error.

It can read JSON response data from files and populate a dictionary of responses based on the requested URL and method.
The server can handle incoming requests by matching the URL and method
to the registered callbacks or the predefined responses.
If a requested endpoint or method is not found, the server returns a 404 Not Found response.
The server can also handle cases where the requested resource cannot generate a representation that
corresponds to the specified Except header, returning a 406 Not Acceptable response.
The class provides methods to set and unset the flag for simulating an HTTP 500 error.
Overall, the MockServer class allows for the creation of a mock server that emulates the behavior of
an actual server, enabling testing and development in a controlled environment.

Author: Mus mbayramo@stanford.edu

"""
import argparse
import json
import os
from random import random
from typing import Callable, Any, Dict, Optional

import requests

from igc.interfaces.rest_mapping_interface import RestMappingInterface
from igc.modules.base.igc_abstract_logger import AbstractLogger
import random


class MockResponse:
    """
    Represents a mock HTTP response.
    """

    def __init__(self, json_data, status_code, error=False, new_state=None):
        """
        Initialize the MockResponse object.

        Both live and mock request encapsulated,  Agent see this.

        :param json_data: JSON data of the response.
        :param status_code: Status code of the response.
        :param error: Indicates if the response represents an error.
        :param new_state: New state information.
        """
        self.json_data = json_data
        self.status_code = status_code
        self.error = error
        self.new_state = new_state

    def json(self):
        """
        Return the JSON data of the response.

        :return: JSON data.
        """
        return self.json_data

    def state(self):
        """
        Return the new state information.

        :return: New state information.
        """
        return self.new_state

    def __str__(self):
        """
        Return a string representation of the MockResponse.

        :return: String representation.
        """
        return f"MockResponse(status_" \
               f"code={self.status_code}, " \
               f"error={self.error}, " \
               f"new_state={self.new_state})"


class MockErrors:
    """

    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._error_response = None

    def generate_error_response(self):
        """
        :return:
        """
        if self._error_response is None:
            error_response = {
                "error": {
                    "@Message.ExtendedInfo": [
                        {
                            "Message": "Unable to complete the operation because the JSON data format entered is invalid.",
                            "MessageArgs": [],
                            "MessageArgs@odata.count": 0,
                            "MessageId": "IDRAC.1.6.SYS405",
                            "RelatedProperties": [],
                            "RelatedProperties@odata.count": 0,
                            "Resolution": "Do the following and the retry the operation: "
                                          "1) Enter the correct JSON data format and retry the operation.",
                            "Severity": "Critical"
                        },
                        {
                            "Message": "The request body submitted was malformed JSON "
                                       "and could not be parsed by the receiving service.",
                            "MessageArgs": [],
                            "MessageArgs@odata.count": 0,
                            "MessageId": "Base.1.2.MalformedJSON",
                            "RelatedProperties": [],
                            "RelatedProperties@odata.count": 0,
                            "Resolution": "Ensure that the request body is valid JSON and resubmit the request.",
                            "Severity": "Critical"
                        }
                    ],
                    "code": "Base.1.2.GeneralError",
                    "message": "A general error has occurred. See ExtendedInfo for more information"
                }
            }
            self._error_response = json.dumps(error_response)

        return self._error_response


class MockServer:
    """
    """
    mock_errors = MockErrors()
    error_response = MockResponse(mock_errors.generate_error_response(), 400, error=True)

    def __init__(self,
                 args: argparse.Namespace,
                 rest_mapping: RestMappingInterface = None,
                 redfish_ip: Optional[str] = "",
                 redfish_username: Optional[str] = "root",
                 redfish_password: Optional[str] = "",
                 redfish_port: Optional[int] = 443,
                 insecure: Optional[bool] = False,
                 is_http: Optional[bool] = False,
                 x_auth: Optional[str] = None,
                 is_live: Optional[bool] = False
                 ):
        """Initialize the MockServer object.
        :param args:
        """

        # flag to generate error 500
        self._is_live = False
        self._is_collect_all = False
        self._is_error_500 = False
        self._valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'PATCH', "HEAD"]

        if not isinstance(args, argparse.Namespace):
            raise TypeError("Invalid args type. Expected argparse.Namespace.")

        if not is_live:
            if not isinstance(rest_mapping, RestMappingInterface):
                raise TypeError("Invalid rest_mapping type. Expected RestMappingInterface.")

        self.logger = AbstractLogger.create_logger(__name__)
        self.logger.info(f"Loading REST API Mock: "
                         f"rest_mapping={rest_mapping}, "
                         f"redfish_ip={redfish_ip}, "
                         f"redfish_username={redfish_username}, "
                         f"redfish_password={redfish_password}, "
                         f"redfish_port={redfish_port}, "
                         f"insecure={insecure}, "
                         f"is_http={is_http}, "
                         f"x_auth={x_auth}, "
                         f"is_live={is_live}")

        self._rest_mapping = rest_mapping
        self.responses = {}
        self.dir_mock_resp = os.path.expanduser(args.raw_data_dir)
        # Read JSON files and populate the responses dictionary
        self._mock_callbacks = {}

        # this what we expect it might change for different system
        self._default_rest_prefix = "/redfish/v1"
        self._default_success_code = 200

        #  load all the json files
        self._construct_json_mocks()
        self._error_respond = None

        # This for live execution
        self._is_live = is_live
        self._redfish_ip = redfish_ip
        self.content_type = {'Content-Type': 'application/json; charset=utf-8'}
        self.json_content_type = {'Content-Type': 'application/json; charset=utf-8'}
        self._port = redfish_port
        self._is_verify_cert = insecure
        self._x_auth = x_auth
        self._is_http = is_http

        self._password = redfish_password
        self._username = redfish_username

        self._default_method = "https://"
        if self._is_http:
            self._default_method = "http://"

        self.error_dir = os.path.join(args.raw_data_dir, "all_errors")
        os.makedirs(self.error_dir, exist_ok=True)

        self.observation_dir = os.path.join(args.raw_data_dir, "observation")
        os.makedirs(self.observation_dir, exist_ok=True)

    def is_live_req(self):
        return self._is_live

    @property
    def redfish_ip(self) -> str:
        """redfish port extractor
        :return:
        """
        if ":" in self._redfish_ip:
            return self._redfish_ip
        else:
            if self._port != 443:
                return f"{self._redfish_ip}:{self._port}"
            else:
                return self._redfish_ip

    @property
    def username(self) -> str:
        return self._username

    @property
    def password(self) -> str:
        return self._password

    @property
    def x_auth(self) -> str:
        return self._x_auth

    def authentication_header(self):
        pass

    def _api_head_call(
        self,
        req: str,
        hdr: dict
    ) -> requests.models.Response:
        """Make HTTP HEAD request.
        :param req: path to a path request
        :param hdr: header that will append.
        :return: response.
        """
        headers = {}
        headers.update(self.content_type)
        if hdr is not None:
            headers.update(hdr)

        full_url = self.redfish_ip + req
        self.logger.debug(f"GET Request URL: {full_url} "
                          f"username {self.username} "
                          f"password {self.password}")
        self.logger.debug(f"GET Request Headers: {headers}")

        if self.x_auth is not None:
            headers.update({'X-Auth-Token': self.x_auth})
            return requests.head(
                full_url,
                verify=self._is_verify_cert,
                headers=headers
            )
        else:
            headers.update({'Authorization': f"{self.username}:{self.password}"})
            self.logger.debug(f"GET Request Headers (Authorization): {headers}")

            return requests.head(
                full_url,
                verify=self._is_verify_cert,
                headers=headers,
                auth=(self.username, self.password)
            )

    def _api_delete_call(
        self,
        req: str,
        hdr: dict
    ) -> requests.models.Response:
        """Make HTTP DELETE request.
        :param req: path to a path request
        :param hdr: header that will append.
        :return: response.
        """
        headers = {}
        headers.update(self.content_type)
        if hdr is not None:
            headers.update(hdr)

        full_url = self.redfish_ip + req
        self.logger.debug(f"GET Request URL: {full_url} "
                          f"username {self.username} "
                          f"password {self.password}")
        self.logger.debug(f"GET Request Headers: {headers}")

        if self.x_auth is not None:
            headers.update({'X-Auth-Token': self.x_auth})
            return requests.delete(
                full_url,
                verify=self._is_verify_cert,
                headers=headers
            )
        else:
            headers.update({'Authorization': f"{self.username}:{self.password}"})
            self.logger.debug(f"GET Request Headers (Authorization): {headers}")
            return requests.delete(
                full_url,
                verify=self._is_verify_cert,
                headers=headers,
                auth=(self.username, self.password)
            )

    def _api_post_call(
        self, req: str,
        payload: str,
        hdr: dict
    ) -> requests.models.Response:
        """Make HTTP post request.
        :param req: path to a path request
        :param payload:  json payload
        :param hdr: header that will append.
        :return: response.
        """
        headers = {}
        headers.update(self.content_type)
        if hdr is not None:
            headers.update(hdr)

        full_url = self.redfish_ip + req
        self.logger.debug(f"GET Request URL: {full_url} "
                          f"username {self.username} "
                          f"password {self.password}")
        self.logger.debug(f"GET Request Headers: {headers}")

        if self.x_auth is not None:
            headers.update({'X-Auth-Token': self.x_auth})
            return requests.post(
                full_url,
                data=payload,
                verify=self._is_verify_cert,
                headers=headers
            )
        else:
            headers.update({'Authorization': f"{self.username}:{self.password}"})
            self.logger.debug(f"GET Request Headers (Authorization): {headers}")
            return requests.post(
                full_url,
                data=payload,
                verify=self._is_verify_cert,
                headers=headers,
                auth=(self.username, self.password)
            )

    def _api_patch_call(
        self,
        req: str,
        payload: str,
        hdr: dict
    ) -> requests.models.Response:
        """Make api patch request.
        :param req: path to a path request
        :param payload: json payload
        :param hdr: header that will append.
        :return: response.
        """
        headers = {}
        headers.update(self.content_type)
        if hdr is not None:
            headers.update(hdr)

        full_url = self.redfish_ip + req
        self.logger.debug(f"GET Request URL: {full_url} "
                          f"username {self.username} "
                          f"password {self.password}")
        self.logger.debug(f"GET Request Headers: {headers}")

        if self.x_auth is not None:
            headers.update({'X-Auth-Token': self.x_auth})
            return requests.patch(
                full_url,
                data=payload,
                verify=self._is_verify_cert,
                headers=headers
            )
        else:
            headers.update({'Authorization': f"{self.username}:{self.password}"})
            self.logger.debug(f"GET Request Headers (Authorization): {headers}")
            return requests.patch(
                full_url, data=payload,
                verify=self._is_verify_cert,
                headers=headers,
                auth=(self.username, self.password)
            )

    def _api_get_call(
        self,
        req: str,
        hdr: dict
    ) -> requests.models.Response:
        """Make HTTP GET request.

        :param req: Path to a path request.
        :param hdr: Headers to append.
        :return: Response.
        """
        headers = {}
        headers.update(self.content_type)

        if hdr is not None:
            headers.update(hdr)

        full_url = self.redfish_ip + req
        self.logger.debug(f"GET Request URL: {full_url} "
                          f"username {self.username} "
                          f"password {self.password}")
        self.logger.debug(f"GET Request Headers: {headers}")

        if self.x_auth is not None:
            headers.update({'X-Auth-Token': self.x_auth})
            return requests.get(
                full_url,
                verify=self._is_verify_cert,
                headers=headers
            )
        else:
            self.logger.info("")
            headers.update({'Authorization': f"{self.username}:{self.password}"})
            self.logger.debug(f"GET Request Headers (Authorization): {headers}")
            return requests.get(
                full_url,
                verify=self._is_verify_cert,
                headers=headers,
                auth=(self.username, self.password)
            )

    @staticmethod
    def generate_error_response():
        return MockServer.mock_errors.generate_error_response()

    def register_callback(self, url: str, method: str, callback):
        """Register a callback for a specific endpoint and method.
          It might positive or something we expect or not.

        :param url: The URL of the endpoint.
        :param method: The HTTP method.
        :param callback: The callback function.
        """
        self._mock_callbacks[(url, method)] = callback

    def generic_error_response(self):
        """Return a generic error response.
        :return:
        """
        if self._error_respond is None:
            self._error_respond = MockServer.generate_error_response()

        return self._error_respond

    def _load_responses(self, file_path, file_name):
        """
        Load json file from a file and convert file back to rest api
        this method we essentially take _redfish_v1_UpdateService_FirmwareInventory.json
        and convert back /redfish/v1/UpdateService/FirmwareInventory
        The preferred way we use RestMappingInterface

        :param file_name:
        :param file_path:
        :return:
        """
        # this a quick one later need optimize
        if file_name.endswith('.json'):
            # file_path = os.path.join(root, file)
            url = file_path.replace(self.dir_mock_resp, '')
            url = url.replace(os.sep, '/', 1)
            url = url.replace('_', '/').replace('.json', '')
            url = url.replace('//', '/')
            with open(file_path, 'r') as f:
                json_data = f.read()

            self.add_response(url, 'GET', json_data, self._default_success_code)
            self.add_response(url, 'HEAD', None, self._default_success_code)

    def _load_responses_from_path(self, rest_api_uri: str, file_path: str):
        """
        Load json file from a file and convert file back to rest api.
        This method essentially takes _redfish_v1_UpdateService_FirmwareInventory.json
        and converts it back to /redfish/v1/UpdateService/FirmwareInventory.
        The preferred way is to use RestMappingInterface.

        :param rest_api_uri: The REST API URI.
        :param file_path: The file path.
        :return:
        """
        if not isinstance(rest_api_uri, str):
            raise TypeError("Invalid rest_api_uri type. Expected str.")

        if not isinstance(file_path, str):
            raise TypeError("Invalid file_path type. Expected str.")

        if not rest_api_uri.startswith(self._default_rest_prefix):
            raise ValueError(f"Invalid rest_api_uri format. Must start with {self._default_rest_prefix}.")

        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                json_data = f.read()
            # We read and add the API response.
            self.add_response(rest_api_uri, 'GET', json_data, self._default_success_code)

    def _construct_json_mocks(self) -> None:
        """Construct rest api mock from either from _rest_mapping provided during
        construction or by scanning directory.
        :return:
        """
        if self._rest_mapping is not None:
            for rest_api, path in self._rest_mapping.get_rest_api_mappings():
                self._load_responses_from_path(rest_api, path)
        else:
            for root, dirs, files in os.walk(self.dir_mock_resp):
                for file_name in files:
                    if file_name.endswith(".json"):
                        file_path = os.path.join(root, file_name)
                        self._load_responses(file_path, file_name)

    def _read_json_file(self, url):
        """Load json
        :param url:
        :return:
        """
        file_path = os.path.join(self.dir_mock_resp, url.replace('/', '_') + '.json')
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                json_data = file.read()
            return json_data
        else:
            return None

    def add_response(self, url, method, json_data, status_code):
        """
        :param url:
        :param method:
        :param json_data:
        :param status_code:
        :return:
        """

        self.responses[(url, method)] = {
            "json_data": json_data,
            "status_code": status_code
        }

        # by default, we return 404
        if method != 'GET':
            self.responses[(url, method)] = {
                "json_data": None,
                "status_code": 404
            }

    def callback_dispatcher(
        self,
        callback: Callable[[Any, Dict[str, Any]], MockResponse],
        url: str,
        json_data: Optional[str]
    ) -> MockResponse:
        """
         Dispatches the callback function and handles the response.

        :param callback: The callback function to be executed.
        :param url: The URL of the request.
        :param json_data: The JSON data received in the request.
        :return: The mock response returned by the callback.
        """
        respond = None
        try:
            handler_view = self.responses.get((url, "GET"))
            respond = callback(json_data, handler_view)
            if respond.state() is not None:
                handler_view["json_data"] = json.dumps(respond.state(), indent=4)
        except Exception as e:
            print(e)

        return respond

    def _save_error_response_to_file(self, url, response):
        """
        Save the JSON response to a separate file.

        :param url: URL of the request.
        :param response: Response object containing the JSON data.
        """
        filename = url.replace('/', '_') + f"_{random.randint(1, 100000)}.json"
        error_dirs = f"{self.dir_mock_resp}/all_errors"
        os.makedirs(error_dirs, exist_ok=True)

        filepath = os.path.join(error_dirs, filename)

        try:
            json_data = response.json()
            with open(filepath, 'w') as file:
                json.dump(json_data, file, indent=4)
        except ValueError:
            # saving raw data
            with open(filepath, 'wb') as file:
                file.write(response.content)

    def _save_observation_to_file(self, url, response):
        """
        Save all good observation, so we collect and update dateset

        :param url: URL of the request.
        :param response: Response object containing the JSON data.
        """
        filename = url.replace('/', '_') + f"_{random.randint(1, 100000)}.json"
        filepath = os.path.join(self.observation_dir, filename)

        try:
            json_data = response.json()
            with open(filepath, 'w') as file:
                json.dump(json_data, file, indent=4)
        except ValueError:
            with open(filepath, 'wb') as file:
                file.write(response.content)

    def request(
        self, url,
        method='GET',
        json_data=None, accept_header=None
    ):
        """
         The main interface to mock server. Agent uses this

         If the MockServer is in live mode, it sends the request using the corresponding API call
        (e.g., api_get_call, api_post_call) based on the method parameter. The response is then
        processed and saved accordingly. If the response has a status code of 299 or higher, it is
        considered an error and saved to the error response directory. Otherwise, if the
        `is_collect_all` flag is set, the response is saved to the observation directory for further
        data collection. Finally, the processed response is returned as a MockResponse object.

        If the MockServer is not in live mode, it checks if the provided method is valid.

        If the
        `json_data` is provided, it attempts to parse it as JSON. If parsing fails, an error response
        is returned. If the `_is_error_500` flag is set, a critical error response is returned.

        The MockServer then checks for a callback corresponding to the provided URL and method,
        and if found, it dispatches the callback function. If not, it checks if there is a pre-defined
        response for the provided URL and method, and if found, returns the corresponding response.
        If no response is found, it checks if there is a matching endpoint URL with a different method,
        and returns a "Method Not Allowed" error response. If the `accept_header` is specified and
        the requested resource cannot generate a representation that matches the Accept header,
        a "Not Acceptable" error response is returned. If none of the above conditions match,
        a "Not Found" error response is returned.


        :param url: The URL of the request.
        :param method: The HTTP method of the request.
        :param json_data: The JSON data of the request.
        :param accept_header: The Accept header of the request.
        :return: A MockResponse object representing the response.
        """

        if self._is_live:
            if method == 'GET':
                response = self._api_get_call(url, hdr={'Accept': accept_header})
            elif method == 'POST':
                response = self._api_post_call(url, json_data, hdr={'Accept': accept_header})
            elif method == 'PATCH':
                response = self._api_patch_call(url, json_data, hdr={'Accept': accept_header})
            elif method == 'HEAD':
                response = self._api_head_call(url, hdr={'Accept': accept_header})
                return MockResponse(response, response.status_code)
            elif method == 'DELETE':
                response = self._api_delete_call(url, hdr={'Accept': accept_header})
            else:
                return MockResponse(None, 405, error=True)

            # we collect all errors and all observation
            if response.status_code >= 299:
                self._save_error_response_to_file(url, response)
            else:
                # in case we want collect more data.
                if self._is_collect_all:
                    self._save_observation_to_file(url, response)

            return MockResponse(response.json(), response.status_code)

        if method not in self._valid_methods:
            return MockServer.error_response

        if json_data is not None:
            try:
                json.loads(json_data)
            except json.JSONDecodeError:
                return MockServer.error_response

        # generate critical errors if flag set
        if self._is_error_500:
            return MockResponse(
                MockServer.generate_error_response(), 500, error=True)

        # dispatch
        callback = self._mock_callbacks.get((url, method))
        if callback:
            return self.callback_dispatcher(callback, url, json_data)

        # get handler
        response_data = self.responses.get((url, method))
        if response_data:
            return MockResponse(response_data["json_data"],
                                response_data["status_code"])
        else:
            for endpoint, data in self.responses.items():
                endpoint_url, endpoint_method = endpoint
                if endpoint_url == url and endpoint_method != method:
                    return MockResponse(MockServer.generate_error_response(), 405, error=True)

            # check if the Accept header is specified and the requested resource cannot generate
            # a representation that corresponds to one of the media types in the Accept header
            if accept_header is not None:
                return MockResponse(MockServer.generate_error_response(), 406, error=True)

            # Return 404 Not Found if the endpoint doesn't exist
            return MockResponse(MockServer.generate_error_response(), 404, error=True)

    def set_simulate_http_500_error(self, val: bool):
        """Set the flag to simulate HTTP 500 error.
        :param val:
        :return:
        """
        self._is_error_500 = val
