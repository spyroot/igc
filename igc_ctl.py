import argparse
import logging
import os
import ssl
import sys
import warnings

import requests
import urllib3

from idrac_ctl import IDracManager, CustomArgumentDefaultsHelpFormatter, ApiRequestType
from idrac_ctl.build.lib.idrac_ctl.idrac_main import create_cmd_tree
from idrac_ctl.idrac_ctl.cmd_exceptions import AuthenticationFailed
from idrac_ctl.idrac_ctl.idrac_main import console_error_printer

try:
    from urllib3.exceptions import InsecureRequestWarning

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError as ir:
    warnings.warn("Failed import urllib3")

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s '
                           '[%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.ERROR)

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
logger.addHandler(console_handler)


class TermColors:
    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_CYAN = '\033[96m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


"note we do sub-string match"
TermList = ["xterm", "linux", "ansi", "xterm-256color"]


def collect_schemas(cmd_args):
    """Invoke discovery action and collection all responses.
    """
    redfish_api = IDracManager(
        idrac_ip=cmd_args.idrac_ip,
        idrac_username=cmd_args.idrac_username,
        idrac_password=cmd_args.idrac_password,
        idrac_port=cmd_args.idrac_port,
        insecure=cmd_args.insecure,
        is_http=cmd_args.use_http,
        is_debug=cmd_args.debug)
    _ = redfish_api.check_api_version()

    system_state = redfish_api.sync_invoke(
        ApiRequestType.Discovery, "discovery"
    )


def igc_main_ctl():
    """
    """
    logger.setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(
        prog="igc_ctl", add_help=True,
        description='''IGC Discover offline tools. |n
                                     It a standalone command line tool provide option to interact with  |n 
                                     Dell iDRAC via Redfish REST API. It supports both asynchronous and |n
                                     synchronous options to interact with iDRAC.|n
                                     Author Mus''',
        epilog='''For more detail, for example, documentation. Make sure to check.
                                             https://github.com/spyroot/idrac_ctl |n
                                             The example folder container many examples.
                                             Author Mustafa Bayramov spyroot@gmail.com
                                             ''',
        formatter_class=CustomArgumentDefaultsHelpFormatter)

    credentials = parser.add_argument_group('credentials', '# igc credentials details.')

    # global args
    credentials.add_argument(
        '--idrac_ip', required=False, type=str,
        default=os.environ.get('IDRAC_IP', ''),
        help="idrac ip address, by default "
             "read from environment IDRAC_IP.")

    credentials.add_argument(
        '--idrac_username', required=False, type=str,
        default=os.environ.get('IDRAC_USERNAME', 'root'),
        help="idrac ip address, by default "
             "read from environment IDRAC_USERNAME.")
    credentials.add_argument(

        '--idrac_password', required=False, type=str,
        default=os.environ.get('IDRAC_PASSWORD', ''),
        help="idrac ip address, by default "
             "read from environment IDRAC_PASSWORD.")

    credentials.add_argument(
        '--idrac_port', required=False, type=int,
        default=int(os.environ.get('IDRAC_PORT', 443)),
        help="idrac port address, by default "
             "read from environment IDRAC_PORT.")

    credentials.add_argument(
        '--insecure', action='store_true', required=False,
        help="insecure ssl.")

    credentials.add_argument(
        '--use_http', action='store_true', required=False, default=False,
        help="use http instead https as a transport.")

    verbose_group = parser.add_argument_group(
        'verbose', '# verbose and debug options'
    )

    verbose_group.add_argument(
        '--debug', action='store_true', required=False,
        help="enables debug.")

    verbose_group.add_argument(
        '--verbose', action='store_true', required=False, default=False,
        help="enables verbose output.")

    verbose_group.add_argument(
        '--log', required=False, default=logging.NOTSET,
        help="log level.")

    # controls for output
    output_controllers = parser.add_argument_group('output', '# output controller options')
    output_controllers.add_argument(
        '--no_extra', action='store_true', required=False, default=False,
        help="disables extra data stdout output.")

    output_controllers.add_argument(
        '--no_action', action='store_true', required=False, default=False,
        help="disables rest action data stdout output.")

    output_controllers.add_argument(
        '--json', action='store_true', required=False, default=True,
        help="by default we use json to output to console.")

    output_controllers.add_argument(

        '--json_only', action='store_true', required=False, default=False,
        help="by default output has different section. "
             "--json_only will merge all in one single output.")

    output_controllers.add_argument(
        '-d', '--data_only', action='store_true', required=False, default=False,
        help="for commands where we only need single value from json.")

    output_controllers.add_argument(

        '--no-stdout', '--no_stdout', action='store_true', required=False, default=False,
        help="by default we use stdout output.")

    output_controllers.add_argument(
        '--nocolor', action='store_false', required=False, default=True,
        help="by default output to terminal is colorful.")

    output_controllers.add_argument(
        '-f', '--filename', required=False, type=str,
        default="", help="Filename if we need save to a file.")

    args = parser.parse_args()
    if args.debug:
        logger.setLevel(args.log)

    if args.idrac_ip is None or len(args.idrac_ip) == 0:
        print(
            "Please indicate the idrac ip. "
            "--idrac_ip or set IDRAC_IP environment variable. "
            "(export IDRAC_IP=ip_address)"
        )
        sys.exit(1)
    if args.idrac_username is None or len(args.idrac_username) == 0:
        print(
            "Please indicate the idrac username."
            "--idrac_username or set IDRAC_USERNAME environment variable. "
            "(export IDRAC_USERNAME=ip_address)"
        )
        sys.exit(1)
    if args.idrac_password is None or len(args.idrac_password) == 0:
        print(
            "Please indicate the idrac password. "
            "--idrac_password or set IDRAC_PASSWORD environment."
            "(export IDRAC_PASSWORD=ip_address)"
        )
        sys.exit(1)
    try:
        collect_schemas(args)

    except AuthenticationFailed as af:
        console_error_printer(f"Error: {af}")
    except requests.exceptions.ConnectionError as http_error:
        console_error_printer(f"Error: {http_error}")
    except ssl.SSLCertVerificationError as ssl_err:
        console_error_printer(f"Error: {ssl_err}")


if __name__ == "__main__":
    igc_main_ctl()
