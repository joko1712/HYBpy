import sys
input = sys.stdin.read

def main():
    # Reading input once and splitting into lines
    data = input().split()
    print(data)
    
    results = []
    process_cases(int(data[0]), data, 1, results)
    
    # Store results
    for result in results:
        print(result)

def process_cases(num_cases, data, index, results):
    if num_cases == 0:
        return
    num_integers = int(data[index])
    integers = list(map(int, data[index+1:index+1+num_integers]))
    print(integers)
    # Processing current case using a recursive approach
    sum_squares = sum_squares_positive(integers, 0, 0)
    results.append(sum_squares)
    # Move to next case
    process_cases(num_cases - 1, data, index + 1 + num_integers, results)

def sum_squares_positive(numbers, index, accum):
    if index == len(numbers):
        return accum
    current = numbers[index]
    if current > 0:
        accum += current * current
    return sum_squares_positive(numbers, index + 1, accum)

if __name__ == "__main__":
    main()