import requests

url = 'http://0.0.0.0/predict-batch'
#requestBody = { "texts": ["احمد خالد محمد","اسد محمد غضنفر"]}
print('Welcome in Name Verification ap :)')
user_input = input('Which Name do you want to verify?\n')

requestBody = { "texts": [user_input.strip()]}
x = requests.post(url, json = requestBody)
print(f"Response {(1)} result: \n",x.json())
i=0
while(True):
    user_input = input('Do you want another Name (y/[n])? ')

    if user_input != 'n':
        msg=["Enter the Name you want to test: \n ","Let's do this: \n","One more time: \n","Enjoy: \n"]
        user_input = input(msg[i%len(msg)])
        requestBody = { "texts": [user_input.strip()]}
        x = requests.post(url, json = requestBody)
        print(f"Response {(i+2)} result: \n", x.json())

    else:
        print("Thank you have a nice day :D")
        break
    i+=1
