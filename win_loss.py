import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns


matches = pd.read_csv("C:/Users/Harsha/Prem_Match_Data/df_full_premierleague.csv")

matches = matches.dropna()
del matches['link_match']; del matches['index']; del matches['season']

# Add a new column "winner"
matches["Winner"] = None  # Initialize the "winner" column with None

# Iterate through rows and determine the winner
for index, row in matches.iterrows():
    if row["goal_home_ft"] > row["goal_away_ft"]:
        matches.at[index, "Winner"] = row["home_team"]
    elif row["goal_home_ft"] < row["goal_away_ft"]:
        matches.at[index, "Winner"] = row["away_team"]
    else:
        matches.at[index, "Winner"] = "Draw"


matches["gd"] = matches['goal_home_ft'] - matches['goal_away_ft']

matches["date"] = pd.to_datetime(matches["date"])


# Create a dictionary to map team names to unique numerical codes
team_to_num = {team: code for code, team in enumerate(matches['home_team'].unique(), start=1)}

# Add new columns with numerical codes for home and away teams
matches['home_team_code'] = matches['home_team'].map(team_to_num)
matches['away_team_code'] = matches['away_team'].map(team_to_num)

# Convert the winner column to numerical values (team codes) or "Draw" as numbers
matches['winner_code'] = matches['Winner'].apply(lambda winner: team_to_num.get(winner, 0))

# Initialize empty lists to hold data
all_matches = []

# Iterate through each row and create match entries for both home and away
for index, row in matches.iterrows():
    home_match = {
        'date': row['date'],
        'team_code': row['home_team_code'],
        'team_name': row['home_team'],
        'opponent_code': row['away_team_code'],
        'opponent_name': row['away_team'],
        'is_home': True,
        'goal_for': row['goal_home_ft'],
        'goal_against': row['goal_away_ft'],
        'winner': row['Winner'],  
        'win_code': row['winner_code'],
        'gd': row['gd'],
        'possession_avg': row['possession_avg_home'],
        'shots_avg': row['shots_avg_home'],
        'shots_on_target_avg': row['shots_on_target_avg_home'],
        'clearances_avg': row['clearances_avg_home'],
        'corners_avg': row['corners_avg_home'],
        'fouls_conceded_avg': row['fouls_conceded_avg_home'],
        'offsides_avg': row['offsides_avg_home'],
        'passes_avg': row['passes_avg_home']
    }
    away_match = {
        'date': row['date'],
        'team_code': row['away_team_code'],
        'team_name': row['away_team'],
        'opponent_code': row['home_team_code'],
        'opponent_name': row['home_team'],
        'is_home': False,
        'goal_for': row['goal_away_ft'],
        'goal_against': row['goal_home_ft'],
        'winner': row['Winner'],  
        'win_code': row['winner_code'],
        'gd': -row['gd'],
        'possession_avg': row['possession_avg_away'],
        'shots_avg': row['shots_avg_away'],
        'shots_on_target_avg': row['shots_on_target_avg_away'],
        'clearances_avg': row['clearances_avg_away'],
        'corners_avg': row['corners_avg_away'],
        'fouls_conceded_avg': row['fouls_conceded_avg_away'],
        'offsides_avg': row['offsides_avg_away'],
        'passes_avg': row['passes_avg_away']
    }
    
    all_matches.append(home_match)
    all_matches.append(away_match)

# Create a DataFrame from the list of dictionaries
all_matches_df = pd.DataFrame(all_matches)

# Sort the DataFrame by team code and date
all_matches_df.sort_values(by=['team_code', 'date'], inplace=True)

# Convert 'is_home' column to 1 for True and 0 for False
all_matches_df['is_home'] = all_matches_df['is_home'].astype(int)

# Now all_matches_df is organized by each team's schedule
print(all_matches_df[['date','team_name','team_code','opponent_name','opponent_code','is_home', 'winner', 'gd']])

print(all_matches_df.head())


# Specify the features 
features = ['possession_avg', 'possession_avg','shots_avg', 
            'shots_on_target_avg', 'clearances_avg','corners_avg', 
            'fouls_conceded_avg', 'offsides_avg', 'passes_avg', 
            'team_code', 'opponent_code', 'is_home']

# Specify the features for which you want to calculate rolling averages
features_to_average = ['gd']

# Calculate rolling averages for each feature and add to the DataFrame
prev_games = 3
for feature in features_to_average:
    rolling_avg = all_matches_df.groupby('team_code')[feature].rolling(window=prev_games, min_periods=1, closed='left').mean()
    all_matches_df[f'{feature}_rolling_avg'] = rolling_avg.reset_index(drop=True)


# Display the DataFrame with rolling averages
print(all_matches_df[['date', 'team_name', 'team_code', 'opponent_name', 'opponent_code', 'is_home', 'winner', 'gd', 'gd_rolling_avg']])

# Fill NaN values using forward fill method
all_matches_df.fillna(method='ffill', inplace=True)

def get_result(row):
    if row["winner"] == "Draw":
        return "Draw"
    elif row["winner"] == row["team_name"]:
        return "Win"
    else:
        return "Loss"

all_matches_df["result"] = all_matches_df.apply(get_result, axis=1)

# Create a dictionary to map result names to numerical codes
result_to_code = {"Win": 1, "Draw": 0, "Loss": -1}

# Add a new column "result_code" based on the mapping
all_matches_df["result_code"] = all_matches_df["result"].map(result_to_code)

# Display the DataFrame with the "result" column
print(all_matches_df[['date', 'team_name', 'opponent_name', 'is_home', 'winner', 'result', 'result_code']])

# Remove rows with the "Draw" result
all_matches_df = all_matches_df.query('result != "Draw"')


# Split data into training and testing sets based on the split date
split_date = pd.to_datetime("2020-08-11")
train_data = all_matches_df[all_matches_df["date"] < split_date]
test_data = all_matches_df[all_matches_df["date"] >= split_date]

# Define the target variable (result_code)
target = 'result_code'

f = ['possession_avg', 'possession_avg', 'gd_rolling_avg',
    'shots_avg', 'shots_on_target_avg', 'clearances_avg',
    'corners_avg','fouls_conceded_avg', 
    'offsides_avg', 'passes_avg', 
    'team_code', 'opponent_code', 'is_home']


# Create the model
model = RandomForestClassifier(n_estimators=300, min_samples_split=50, random_state=4)

# Train the model
model.fit(train_data[f], train_data[target])

# Make predictions on the test data
pred = model.predict(test_data[f])

# Calculate accuracy
accuracy = accuracy_score(test_data[target], pred)
print("Accuracy:", accuracy)

# Create a comparison DataFrame
comparison_df = pd.DataFrame({
    "date": test_data["date"],
    "team_name": test_data["team_name"],
    "team_code": test_data["team_code"],
    "opponent_name": test_data["opponent_name"],
    "opponent_code": test_data["opponent_code"],
    "is_home": test_data['is_home'],
    "actual_result": test_data["result"],
    "actual_result_code": test_data[target],
    "predicted_result_code": pred
})

result_labels = {1: "Win", -1: "Loss"}
comparison_df["Result"] = comparison_df["predicted_result_code"].map(result_labels)

# Display the updated DataFrame
print(comparison_df)

# Calculate accuracy for different scenarios
overall_accuracy = accuracy_score(comparison_df["actual_result_code"], comparison_df["predicted_result_code"])
win_accuracy = accuracy_score(comparison_df[comparison_df["actual_result_code"] == 1]["actual_result_code"],
                              comparison_df[comparison_df["actual_result_code"] == 1]["predicted_result_code"])
loss_accuracy = accuracy_score(comparison_df[comparison_df["actual_result_code"] == -1]["actual_result_code"],
                               comparison_df[comparison_df["actual_result_code"] == -1]["predicted_result_code"])

print("Overall Accuracy:", overall_accuracy)
print("Win Accuracy:", win_accuracy)
print("Loss Accuracy:", loss_accuracy)

# Calculate True Positives
true_positives = len(comparison_df[(comparison_df['actual_result_code'] == 1) & (comparison_df['predicted_result_code'] == 1)])

# Calculate False Positives
false_positives = len(comparison_df[(comparison_df['actual_result_code'] != 1) & (comparison_df['predicted_result_code'] == 1)])

# Calculate True Positives
true_negatives = len(comparison_df[(comparison_df['actual_result_code'] == -1) & (comparison_df['predicted_result_code'] == -1)])

# Calculate False Positives
false_negatives = len(comparison_df[(comparison_df['actual_result_code'] != -1) & (comparison_df['predicted_result_code'] == -1)])

# Calculate Precision
w_precision = true_positives / (true_positives + false_positives)
l_precision = true_negatives / (true_negatives + false_negatives)

print("Win Precision:", w_precision)
print("Loss Precision:", l_precision)

# Display the comparison DataFrame
print(comparison_df)

# Count the number of wins and losses for each team
team_results = comparison_df.groupby(["team_name", "Result"]).size().unstack(fill_value=0)

# Setting custom colors for "Loss" and "Win"
colors = ["red", "green"]

# Plotting using Seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# Bar plot for wins and losses with custom colors and stacking
team_results.plot(kind="bar", stacked=True, ax=plt.gca(), color=colors)
plt.title("Wins and Losses by Team")
plt.xlabel("Team")
plt.ylabel("Number of Games")
plt.xticks(rotation=45, ha="right")

plt.show()


# Calculate the number of predicted wins for each team
team_predicted_win_counts = comparison_df[comparison_df["predicted_result_code"] == 1].groupby("team_name").size()

# Calculate the total number of games played by each team
team_game_counts = comparison_df.groupby("team_name").size()

# Calculate predicted win ratios for each team
team_predicted_win_ratios = team_predicted_win_counts / team_game_counts

# Create a new DataFrame with team name and predicted win ratio
predicted_team_win_df = pd.DataFrame({
    "Team Name": team_predicted_win_ratios.index,
    "Predicted Win Ratio": team_predicted_win_ratios
})

# Sort the DataFrame by highest to lowest win ratio
predicted_team_win_df = predicted_team_win_df.sort_values(by="Predicted Win Ratio", ascending=False)

# Reset the index and add 1 to start index from 1
predicted_team_win_df.reset_index(drop=True, inplace=True)
predicted_team_win_df.index = predicted_team_win_df.index + 1

# Display the index, team name, and predicted win ratio
print(predicted_team_win_df)











