import random
import json


def calculate_digit_sum(num: int):
    return sum(int(d) for d in str(abs(num)))


def generate_sorting_dataset(size: int, min_length: int = 3, max_length: int = 10, max_num: int = 100):
    dataset = []
    variations = [
        ("Sort ascending: ", lambda x: sorted(x)),
        ("Sort descending: ", lambda x: sorted(x, reverse=True)),
        ("Sort even-odd: ", lambda x: sorted([n for n in x if abs(n) % 2 == 0]) + sorted([n for n in x if abs(n) % 2 == 1])),
        ("Sort odd-even: ", lambda x: sorted([n for n in x if abs(n) % 2 == 1]) + sorted([n for n in x if abs(n) % 2 == 0])),
        ("Sort by last digit: ", lambda x: sorted(x, key=lambda n: abs(n) % 10)),
        ("Sort by digit sum: ", lambda x: sorted(x, key=calculate_digit_sum)),
    ]
    
    for _ in range(size):
        length = random.randint(min_length, max_length)
        numbers = [random.randint(-max_num, max_num) for _ in range(length)]
        
        # Randomly pick a variation
        prefix, sort_func = random.choice(variations)
        input_str = f"{prefix}{', '.join(map(str, numbers))}"
        output_str = ", ".join(map(str, sort_func(numbers)))

        dataset.append({"task": input_str, "ground_truth": output_str})
    return dataset



# Generate dataset
dataset = generate_sorting_dataset(size=200000)

# Print first 5 examples
for i in range(5):
    print(f"Input: {dataset[i]['task']}, Output: {dataset[i]['ground_truth']}")


result = [json.dumps(record) for record in dataset]
with open('math_tasks.jsonl', 'w') as obj:
    for i in result:
        obj.write(i+'\n')