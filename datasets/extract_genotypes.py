import os
import json

def extract_genotypes(root_dir, output_file):
    count = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                
                if filename.endswith('.json'):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if "population" in data:
                                for item in data["population"]:
                                    if "genotype" in item:
                                        geno = item["genotype"].strip()
                                        out_f.write(geno + '\n')
                                        count += 1
                    except Exception as e:
                        print(f"Error reading JSON {filepath}: {e}")
                
                elif filename.endswith('.gen'):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.startswith("genotype:"):
                                    geno = line.split("genotype:", 1)[1].strip()
                                    if geno.startswith('~') and geno.endswith('~'):
                                        geno = geno[1:-1].strip()
                                    out_f.write(geno + '\n')
                                    count += 1
                    except Exception as e:
                        print(f"Error reading GEN {filepath}: {e}")

    print(f"Extraction complete! {count} genotypes saved to {output_file}")

if __name__ == "__main__":
    root_directory = r"d:\Studia\2stopień\Magisterka\datasets\f1\hof"
    out_file = os.path.join(root_directory, "all_genotypes_extracted.txt")
    extract_genotypes(root_directory, out_file)
