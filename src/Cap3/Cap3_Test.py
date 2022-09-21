import tweepy
import json

def APIS_exam():
    CONSUMER_TOKEN = "2WIxFsFs2AMzj2ecJDv6SHc27"
    CONSUMER_SECRET = "iBfGSSgjr9o1q18bULAp0OBeSN7NqU3wPvFUCYJU8KxZgLILZX"
    ACCESS_TOKEN = '1003920321441927168-0GllZYAGrmKA817B3KxT1y1AQxw9Ob'
    ACCESS_TOKEN_SECRET = 'rvhyp5BvkDgJq6bvoTHnhe8AUlOqgCUS1OidPQ6VbI9Tc'

    auth = tweepy.OAuthHandler(CONSUMER_TOKEN, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    lista_tweets = api.search(q="python")
    lista_json = []
    for tweet in lista_tweets:
        lista_json.append(tweet._json)
    lista_json[0]

def auth():
    auth = tweepy.OAuth1UserHandler(
    consumer_key, consumer_secret, access_token, access_token_secret
    )

    api = tweepy.API(auth)

    public_tweets = api.home_timeline()
    for tweet in public_tweets:
        print(tweet.text)

    # Get the User object for twitter...
    user = api.get_user(screen_name='twitter')

def token():
    import os

    consumer_key = os.environ.get("CONSUMER_KEY")   
    consumer_secret = os.environ.get("CONSUMER_SECRET")

    CONSUMER_KEY = f"1toDaFY1YJ7ZX56LsMOs0P0CR"
    CONSUMER_SECRET = f"N3O8THogZRCgNc2r6SQpOLc4QQ2bAzBmHN6xz8k97mGylFV3hB"
    BEAERTOKEN = f"AAAAAAAAAAAAAAAAAAAAAHrAhAEAAAAA0t%2FGtvOuCf%2BTBr3kK7F9b9GB30c%3DZUoLHceWQqeVOTCIfbjQ2yhxRqFUUx8CT3c53FuG61FoRE9Vbg"
 
    auth = tweepy.OAuth2BearerHandler(BEAERTOKEN)
    api = tweepy.API(auth)

    public_tweets = api.home_timeline()
    for tweet in public_tweets:
        print(tweet.text)

if __name__ == '__main__':
    Test = 'token'

    if Test == 'token':
        token()