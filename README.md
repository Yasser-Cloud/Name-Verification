## 1. Build the Docker image

Make sure you are in the project path where docker-compose.yml is

```console

$ docker-compose build . 

```

## 2. Run the Docker image (starts the container)

```console

$ docker-compose up

```
## 3. Run test script

If you want user input (run test script in [test.py](./test.py)):

```console

$ python test.py

```
## 4. Verify the result

According to a probability threshold of 0.5, real names with high confidence were classified as Real names, if the probability was 0.5 or more, otherwise they would be classified as Fake names (real names with low confidence).

## 5. Close Docker image

```console

$ docker-compose down

```

