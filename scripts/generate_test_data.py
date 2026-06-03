import argparse
import random
import struct
import os

def generate_data(num_vectors, dim, output_file):
    print(f"Generating {num_vectors} vectors of dimension {dim}")
    
    with open(output_file, 'wb') as f:
        # I - unsigned 32 bit integer
        # II - 2 unsigned 32 bit integers
        f.write(struct.pack('II', num_vectors, dim))
        
        # Generate and write the random floats
        for i in range(num_vectors):
            # Create a list of random floats between -1.0 and 1.0
            vector = [random.uniform(-1.0, 1.0) for i in range(dim)]
            
            # f'{dim}f' - pack 'dim' number of floats
            f.write(struct.pack(f'{dim}f', *vector))
            
    print(f"Successfully wrote binary data to {output_file} ({(os.path.getsize(output_file) / 1024):.2f} KB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random vectors for VectorForge testing")
    parser.add_argument("--num", type=int, default=1000, help="Number of vectors to generate")
    parser.add_argument("--dim", type=int, default=128, help="Dimensionality of each vector")
    parser.add_argument("--output", type=str, default="test_data.bin", help="Output binary file path")
    
    args = parser.parse_args()
    generate_data(args.num, args.dim, args.output)
