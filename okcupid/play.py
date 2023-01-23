import pandas as pd

okcupid = pd.read_csv('okcupid.csv')
df = okcupid
# df = df[df.status == 'single']


# To classify: sex, orientation
# print(df.columns)
'''
'age', 'status', 'sex', 'orientation', 'body_type', 'diet', 'drinks',
'drugs', 'education', 'ethnicity', 'height', 'income', 'job',
'last_online', 'location', 'offspring', 'pets', 'religion', 'sign',
'smokes', 'speaks'
'''
# print(df.iloc[0])
# good: age, sex, orientation
# bad: status, body_type, diet

'''
age                                                22
status                                         single
sex                                                 m
orientation                                  straight
body_type                              a little extra
diet                                strictly anything
drinks                                       socially
drugs                                           never
education               working on college/university
ethnicity                                asian, white
height                                           75.0
income                                             -1
job                                    transportation
last_online                          2012-06-28-20-30
location              south san francisco, california
offspring      doesn't have kids, but might want them
pets                        likes dogs and likes cats
religion        agnosticism and very serious about it
sign                                           gemini
smokes                                      sometimes
speaks                                        english
'''
print(df.drinks.value_counts())

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
