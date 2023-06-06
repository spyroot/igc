import json
import os
import argparse
from igc.interfaces.rest_mapping_interface import RestMappingInterface


class MockResponse:
    def __init__(self, json_data, status_code, error=False, new_state=None):
        """
        :param json_data:
        :param status_code:
        :param error:
        """
        self.json_data = json_data
        self.status_code = status_code
        self.error = error
        self.new_state = new_state

    def json(self):
        """
        :return:
        """
        return self.json_data

    def state(self):
        """
        :return:
        """
        return self.new_state



class MockServer:
    """
    """
    def __init__(self, args: argparse.Namespace, rest_mapping: RestMappingInterface = None):
        """Initialize the MockServer object.
        :param args:
        """

        # flag to generate error 500
        self._is_error_500 = False
        self._valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'PATCH', "HEAD"]

        if not isinstance(args, argparse.Namespace):
            raise TypeError("Invalid args type. Expected argparse.Namespace.")

        if not isinstance(rest_mapping, RestMappingInterface):
            raise TypeError("Invalid rest_mapping type. Expected RestMappingInterface.")

        self._rest_mapping = rest_mapping
        self.responses = {}
        self.dir_mock_resp = os.path.expanduser(args.raw_data_dir)
        # Read JSON files and populate the responses dictionary
        self.mock_callbacks = {}

        # this what we expect it might change for different system
        self._default_rest_prefix = "/redfish/v1"
        self._default_success_code = 200

        #  load all the json files
        self._construct_json_mocks()
        self._error_respond = None

    def register_callback(self, url: str, method: str, callback):
        """Register a callback for a specific endpoint and method.
          It might positive or something we expect or not.
        :param url: The URL of the endpoint.
        :param method: The HTTP method.
        :param callback: The callback function.
        """
        self.mock_callbacks[(url, method)] = callback

    @staticmethod
    def generate_error_response():
        """
        :return:
        """
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
        return json.dumps(error_response)

    def generic_error_response(self):
        """
        :return:
        """
        if self._error_respond is None:
            self._error_respond = MockServer.generate_error_response()

        return self._error_respond

    def _load_responses(self, file_path, file_name):
        """Load json file from a file and convert file back to rest api
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
            self.responses[(url, method)] = {"json_data": None, "status_code": 404}

    def request(self, url, method='GET', json_data=None, accept_header=None):
        """Main interface to mock server
        :param url:
        :param method:
        :param json_data:
        :param accept_header:
        :return:
        """

        if method not in self._valid_methods:
            return MockResponse(MockServer.generate_error_response(), 400, error=True)

        if json_data is not None:
            try:
                json.loads(json_data)
            except json.JSONDecodeError:
                return MockResponse(MockServer.generate_error_response(), 400, error=True)

        # generate critical errors if flag set
        if self._is_error_500:
            return MockResponse(MockServer.generate_error_response(), 500, error=True)

        # dispatch
        callback = self.mock_callbacks.get((url, method))
        if callback:
            # handler_view = self.responses.get((url, "GET"))
            return callback(json_data, self.responses.get((url, "GET")))

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
