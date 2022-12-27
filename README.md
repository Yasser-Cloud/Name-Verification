1

## 1. Build the Docker image

2

Make sure you are in the project path where docker-compose.yml is

3

​

4

```console

5

$ docker-compose build . 

6

```

7

## 2. Run the Docker image (starts the container)

8

​

9

```console

10

$ docker-compose up

11

```

12

## 3. Run test script

13

If you want user input (run test script in [test.py](./test.py)):

14

​

15

```console

16

$ python test.py

17

```

18

## 4. Verify the result

19

According to a probability threshold of 0.5, real names with high confidence were classified as Real names, if the probability was 0.5 or more, otherwise they would be classified as Fake names (real names with low confidence).

20

​

21

## 5. Close Docker image

22

​

23

```console

24

$ docker-compose down

25

```

