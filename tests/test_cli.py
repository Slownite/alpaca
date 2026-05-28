from alpaca.cli import build_parser


def test_parser_lists_commands():
    parser = build_parser()
    args = parser.parse_args(["ps"])
    assert args.command == "ps"
