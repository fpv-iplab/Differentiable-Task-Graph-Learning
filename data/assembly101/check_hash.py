import hashlib
import click

def are_files_identical(hash, file):
    # Use a hash function (e.g., SHA-256) to check for identical contents
    def get_file_hash(file_path):
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            # Read the file in chunks to avoid memory issues with large files
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    # Read the hash from the file
    with open(hash, 'r') as f:
        hash = f.read()
    
    # Compare the hash of both files
    return hash == get_file_hash(file)


@click.command()
@click.option('--hash', '-h', required=True, help='Path to the file containing the hash value')
@click.option('--file', '-f', required=True, help='Path to the file to compare')
def main(hash:str, file:str):
    if are_files_identical(hash, file):
        print("The hash are identical.")
    else:
        print("The hash are different.")
        
if __name__ == '__main__':
    main()
