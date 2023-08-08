import os
import shutil
import pydicom
from pydicom.dataset import Dataset
import argparse

def convert_ima_to_dicom(ima_path, dicom_dir):
    # Read IMA file
    with open(ima_path, 'rb') as ima_file:
        ima_data = ima_file.read()

    # Create a DICOM dataset
    dicom_dataset = Dataset()
    dicom_dataset.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
    dicom_dataset.SOPInstanceUID = pydicom.uid.generate_uid()
    # Set other DICOM attributes as needed

    # Save the DICOM file
    dicom_path = os.path.join(dicom_dir, f'{os.path.splitext(os.path.basename(ima_path))[0]}.dcm')
    pydicom.filewriter.write_file(dicom_path, dicom_dataset, ima_data, write_like_original=False)


def convert_ima_dir_to_dicom(ima_dir, dicom_dir):
    if not os.path.exists(dicom_dir):
        os.makedirs(dicom_dir)

    for ima_file in os.listdir(ima_dir):
        if ima_file.lower().endswith('.ima'):
            ima_path = os.path.join(ima_dir, ima_file)
            convert_ima_to_dicom(ima_path, dicom_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="This is a command-line tool for"
                                                 " performing DICOM file to PNG file conversion.")
    parser.add_argument("-i", "--INPUT_DATA_PATH", help="path/to/your/input/dcm/directory", type=str,
                        default="ima_files", required=False)
    parser.add_argument("-o", "--OUTPUT_DATA_PATH", help="path/to/your/output/png/directory", type=str,
                        default="dcm_files", required=False)

    args = parser.parse_args()
    args.output_data_path = os.path.expanduser('~')
    convert_ima_to_dicom(args.INPUT_DATA_PATH, args.OUTPUT_DATA_PATH)


if __name__ == '__main__':
    main()
