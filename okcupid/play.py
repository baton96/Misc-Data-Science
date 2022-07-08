import pandas as pd

df = pd.read_csv('okcupid.csv')
print(df.columns)
print(df.iloc[0])

# df['is_studying'] = df.education.str.contains('working on ')?
# df['is_studying'] = df.job == 'student'?
# default is "working on" or "graduated"?

# ethnicities = ['white', 'asian', 'hispanic / latin', 'black', 'other', 'indian', 'pacific islander']
# for ethnicity in ethnicities:
#    df[ethnicity] = df[ethnicity].replace({True: 1, False: 0})
# df = df.drop(columns=['ethnicity'])

# df['religion_approach'] = df.religion.str.split(' ').str[2:].str.join(' ')
# df.religion = df.religion.str.split(' ').str[0]

# df.diet = df.diet.str.split(' ').str[-1]

# df['sign_approach'] = df.sign.str.split(' ').str[3:].str.join(' ').str.replace('&rsquo;', '')
# df = df.drop(columns=['sign'])

# df = df[df.age < 100]
# df.to_csv('okcupid.csv', index=False)
# print(df.status.value_counts(dropna=False))
