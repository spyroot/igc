import json
import os
import sys

from igc.modules.igc_main import IgcMain
from igc.shared.shared_main import shared_main


def save_spec(cmd, parser_groups):
    """
    Save the parameters as a JSON file.

    :param cmd: The argparse.Namespace object containing the program arguments.
    :param parser_groups: List of tuples (section_name, section_parser).
    """
    if not os.path.isdir(cmd.output_dir):
        print(f"Error: Invalid output directory: {cmd.output_dir}. "
              "Please specify a valid directory.", file=sys.stderr)
        sys.exit(1)

    params = {}
    for section_name, section_parser in parser_groups:
        section_args, _ = section_parser.parse_known_args()
        section_keys = vars(section_args).keys()
        params[section_name] = {key: getattr(section_args, key) for key in section_keys}
        if 'device' not in params[section_name]:
            params[section_name]['device'] = None

        params[section_name]['device'] = str(params[section_name]['device'])

    save_path = os.path.join(cmd.output_dir, "parameters.json")
    print(f"Saved model trainer in {save_path}")
    with open(save_path, "w") as f:
        json.dump(params, f, indent=4)


def main(cmd, parser_groups):
    """
    :return:
    """
    save_spec(cmd, parser_groups)
    igc = IgcMain(cmd)
    igc.run()


if __name__ == '__main__':
    args, groups = shared_main()
    main(args, groups)
