import logging

import click

from evaluation.evaluate import evaluate_metrics


@click.command()
@click.option(
    "--data-folder-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to data folder of CAM dataset.",
)
@click.option(
    "--output-folder-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to folder to save evaluation results.",
)
def main(data_folder_path, output_folder_path):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    evaluate_metrics(data_folder_path, output_folder_path)


if __name__ == "__main__":
    main()
