"""
Example script to load imagenet.
"""

from attr import define
from loguru import logger

from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from packg.paths import get_data_dir
from typedparser import VerboseQuietArgs, TypedParser, add_argument
from ovqa.datasets.imagenet import ImagenetClsMetadata
from ovqa.datasets.interface_metadata import ClsMetadataInterface


@define(slots=False)
class Args(VerboseQuietArgs):
    dataset_split: str = add_argument(shortcut="-s", type=str, help="Dataset split", default="val")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)

    imagenet_dir = get_data_dir() / "imagenet1k"
    logger.info(f"Imagenet directory: {imagenet_dir}")

    meta: ClsMetadataInterface = ImagenetClsMetadata.load_split(args.dataset_split)
    logger.info(f"Metadata object: {meta}")
    logger.info(f"Done")


if __name__ == "__main__":
    main()
