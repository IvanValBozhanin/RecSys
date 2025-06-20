input_path = 'ml-latest-small/ml-100k/u.csv'      # Replace with your actual file path
output_path = 'ml-latest-small/ml-100k/u_csv.csv'

with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
    for line in infile:
        # Skip empty lines
        if line.strip() == '':
            continue
        # Split by whitespace and join by comma
        values = line.strip().split()
        csv_line = ','.join(values)
        outfile.write(csv_line + '\n')

print(f'CSV file saved to {output_path}')