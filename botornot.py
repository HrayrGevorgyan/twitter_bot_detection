from botornotauth import auth_tokens
import tweepy
import pickle
import sys

class bot_or_not:
	def __init__(self):
		auth = tweepy.OAuthHandler(auth_tokens['API'], auth_tokens['SECRET'])
		api = tweepy.API(auth)
		self.user = api.get_user(sys.argv[1])

	def get_user_info(self):
		# Get all relevant information about account
		followers_count = self.user.followers_count
		friends_count = self.user.friends_count
		listed_count = self.user.listed_count
		verified = self.user.verified

		return [followers_count,friends_count,listed_count,verified]

	def make_prediction(self,user_info):
		with open('twitter_decision_tree_model.pkl', 'rb') as model_file:
			model = pickle.load(model_file)
			prediction = model.predict([user_info])[0]

			if(bool(prediction)):
				classification = "bot"
			else:
				classification = "human"

		return classification

	def print_findings(self, user_info,classification):
		print("Twitter user %s's stats:\n\tfollowers: %s\n\tfriends: %s\n\tappearances in lists: %s\n\tis verified: %s"%(sys.argv[1],user_info[0],user_info[1],user_info[2],user_info[3]))
		print("\nOur model classifies this account as a %s"%classification)


if __name__ == "__main__":
	bon = bot_or_not()
	bon.print_findings(bon.get_user_info(),bon.make_prediction(bon.get_user_info()))

