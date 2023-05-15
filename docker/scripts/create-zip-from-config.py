import os
import sys
import json
import zipfile


name = 'quenedi'
zip_name = name + '.zip'


def main():
    with open('quenedi.config.json') as f:
        config = json.load(f)

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        inputs_paths = config['network_paths'].values()
        for path in inputs_paths:
            if path is None:
                continue
            zip_ref.write(path, arcname=os.path.join('inputs/', os.path.basename(path)))

        outputs_paths = config['output_paths']
        for path in outputs_paths:
            if path is None:
                continue
            for folder_name, subfolders, filenames in os.walk(path):
                for filename in filenames:
                    file_path = os.path.join(folder_name, filename)
                    zip_ref.write(file_path)

    zip_ref.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: At least one argument is required.")
        print("Usage: python {name} model_folder".format(name=sys.argv[0]))
        sys.exit(1)

    source = os.path.dirname(os.path.abspath(__file__))
    quetzal_root = os.path.abspath(os.path.join(source, '../../..'))
    os.chdir(os.path.abspath(os.path.join(quetzal_root, sys.argv[1])))
    main()
