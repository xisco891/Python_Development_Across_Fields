# -*- coding: utf-8 -*-


'''

  This List of String Matching Algorithms assume that have some sort of 
  nominal variables in a dataframe. 
  
  For the sake of Understanding Code we will assume "fullname" column 
  contains a set of unique names, namely, names of soccer teams in the German
  Bundesliga. 
  
  This code was likely created while I promoted for the position of 
  Data Scientist at e-quadrat

'''

def levehnstein_matching(name, df):
    
    max_score = 0
    from fuzzywuzzy import fuzz
    
    print("Applying Levehnstain Algorithm")
    
    if name not in df["fullname"].unique():
        
        print("Running similarity Test with Levehnstein Algorithm.")
    
        for i, element in enumerate(df["fullname"].unique()):
                    
            if isinstance(element, str) and isinstance(name, str):
                
                ratio = fuzz.ratio(name, element)
                if ratio > max_score:
                    
                    max_score = ratio
                    team_matched = element
                
    else:
        team_matched = name
            
    print("team_matched:", team_matched)
    
    return team_matched



#### Sequence Matcher... 
    
def sequenceMatcher(string1, string2):
    
    from difflib import SequenceMatcher
    
    ratio = SequenceMatcher(None, string1, string2).ratio()
    
    return ratio

from difflib import SequenceMatcher

if team_name not in team_translation["fullname"].unique():
    
    print("Running similarity Test.")

    for i, team in enumerate(team_translation["fullname"].unique()):
        
        print("Match not found....Computing Similarity test")
        
        print("team:", team)
        print("team_name:", team_name)
        
        if isinstance(team_name, str) and isinstance(team, str):
            
            ratio = SequenceMatcher(None, team, team_name).ratio()
            print("Ratio is:", ratio)
            if ratio > max_score:
                
                max_score = ratio
                print("There is a new max score.")
                pos_list = i
                team_matched = team
            
else:
    print("Match Found")
    pos_list = i
    team_matched = team_name
        
        
print("pos_list:", pos_list)
print("team_matched:", team_matched)

return team_matched
            
    