"""

MockServer checker update password ip and the rest

Author: Mus mbayramo@stanford.edu

"""
import argparse
import logging
from igc.envs.rest_mock_server import MockServer
from igc.shared.shared_main import shared_main


def main(args):
    """"
    """

    logging.basicConfig(level=logging.INFO)

    mock_server = MockServer(
        args,
        redfish_ip=args.redfish_ip,
        redfish_username="root",
        redfish_password="",
        redfish_port=args.redfish_port,
        insecure=args.insecure,
        is_http=args.is_http,
        x_auth=args.x_auth,
        is_live=args.live
    )

    # Set the _is_live flag based on the argument
    mock_server._is_live = args.live

    if mock_server.is_live_req():
        response = mock_server.request("/redfish/v1", method="GET", accept_header="application/json")
        print(f"Live Mode - GET Response: {response.status_code}")
        print(f"Live Mode - GET JSON Data: {response.json()}")

        response = mock_server.request("/redfish/v1", method="HEAD", accept_header="application/json")
        print(f"Live Mode - HEAD Response: {response.status_code}")
        print(f"Live Mode - HEAD Content: {response}")


if __name__ == '__main__':
    args = shared_main()
    main(args)
