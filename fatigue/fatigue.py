
import psycopg2
import re
from collections import Counter
import warnings
from wordcloud import WordCloud
import spacy
from nltk.corpus import stopwords
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models import Phrases
 from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import seaborn as sns
import datetime
from time import mktime
import time
from datetime import datetime, timedelta



conn = psycopg2.connect(dbname='iadpprod', user='s107610', host='iagdcaprod.auiag.corp' ,password=os.environ.get('EDH_PASSWORD'))
cur = conn.cursor()
query1 = """select policy_brand_name, claim_number, claim_id, claim_description , policy_number, claim_loss_time, claim_loss_date , 
 total_cost_exc_excess_exc_gst_amount
,gender_lodge, impact_rating_group from dl_analytics.op19_claims 
where general_nature_of_loss_name ='Collision'  and policy_brand_name like 'NRMA%'"""

cur.execute(query1)
rows = cur.fetchall()




query2="""
SELECT claim_id, string_agg(note_body, ', ') AS claim_note
FROM   ctx.cc_pi_note
where note_retired='0'
GROUP  BY claim_id"""

with psycopg2.connect(dbname='iadpprod', user='s107610', host='iagdcaprod.auiag.corp', password='2ea5032eb2a8') as conn:
    with conn.cursor() as cur:
        cur.itersize = 1000

        cur.execute(query2)

        for row in cursor:
            print(row)

cur.execute(query2)
#rows2 = cur.fetchall()
patterns = ['fatigue', 'tired', 'sleep', 'drowsy', 'drowsiness', ' weary(\.)? ', 'weariness', ' wearied(\.)? ',
            'exhausted from', 'slept', ' tiring(\.)?']
patt_lst = [re.compile(".*({}).*".format(pattern), re.DOTALL | re.IGNORECASE) for pattern in patterns]
neg_patt = re.compile(r'.*tired to.*|.*tiring to.*', re.DOTALL | re.IGNORECASE)

claim_id=[]
claim_note=[]
row2 = cur.fetchone()
while row2 :
    for k in patt_lst:
        if re.match(k, row2[1]) and not re.match(neg_patt, row2[1]):
            claim_id.append(row2[0])
            claim_note.append(row2[1])
   row2 = cur.fetchone()

df = pd.DataFrame(rows, columns=['brand', 'claim_number', 'claim_id', 'claim_description', 'policy_number', 'claim_loss_time', 'claim_loss_date',
                                 'total_cost_exc_excess_exc_gst_amount','gender_lodge', 'impact_rating_group'])

list1=[]
for s,j in enumerate(df['claim_description']):

    for k in patt_lst:
        if re.match(k, j) and not re.match(neg_patt, j):
                list1.append(s)

list1 =list(set(list1))

list2=[]
for s,j in enumerate(df['claim_note']):
    for k in patt_lst:
        if re.match(k, j) :
            if not re.match(neg_patt, j):
                list2.append(s)

df_extracted = df.iloc[list1,]
df_extracted.to_csv('out.csv', index = False)
df_extracted2 = df.iloc[list2,]


#
lst2=[]
check_phrase= re.compile(r'.*tiring(\.)?.*', re.DOTALL | re.IGNORECASE)
for s,j in enumerate(df_extracted['claim_description']):
    if re.match(check_phrase, j):
        print(s,j)
        lst2.append(s)
test=df.iloc[lst2,]


########################### times of the day when the accident happen ##############

def strip_time(x) :
    b =str(x)
    return b[10:13]

df_extracted['hour']= df_extracted['claim_loss_time'].apply(strip_time)
c=Counter(df_extracted.hour)
a= {k: v  for k, v in c.items()}
d= pd.DataFrame.from_dict(a, orient='index')
df['hour']=df['claim_loss_time'].apply(strip_time)
c1=Counter(df.hour)
a2 = {k: v for k, v in c1.items()}
a1 = {k: v / sum(c1.values())*100 for k, v in c1.items()}
d1=pd.DataFrame.from_dict(a1, orient='index' )
d2=pd.DataFrame.from_dict({k: a[k]/a2[k]*100 for k in a.keys() & a2}, orient='index')

hours = pd.merge(d2, d1, left_index=True, right_index=True)
test= hours.reset_index().sort_values(by=['index'])
test.columns = ['hour', 'fatigue_claims', 'all_claims']
test=test.reset_index(drop=True)
row_idx = [22,23]
row_idx.extend(list(range(0,21)))
h2 =test.reindex(row_idx)
h2.hour.astype('int64')

# --- generate Plot

#%matplotlib inline
fig, ax = plt.subplots(1, 1)
ax2 = ax.twinx()

ax.set_xlabel('time')
ax.set_ylabel('total claims (%)')
ax.plot(h2.iloc[:,0], h2.iloc[:,1])
ax2.bar(h2.iloc[:,0], h2.iloc[:,2],color= "skyblue", lw=0, alpha=0.7)

ax2.set_ylabel('proportion of fatigue claims out of all claims (%)')

plt.title("times of the day when the accident occurs")
plt.show()
plt.savefig('accident_time.png')


########################## impact rating group  ####################

impact_all = df.loc[df.impact_rating_group.notnull()]
impact_fatigue = df_extracted.loc[df_extracted.impact_rating_group.notnull()]

c=Counter(impact_fatigue.impact_rating_group)
a= {k: v for k, v in c.items()}
d= pd.DataFrame.from_dict(a, orient='index')

c1=Counter(impact_all.impact_rating_group)
a2 = {k: v for k, v in c1.items()}
d1= pd.DataFrame.from_dict({k: v /sum(Counter(impact_all.impact_rating_group).values())*100  for k, v in c1.items()}, orient='index')

a3= {k: a[k]/a2[k]*100for k in a.keys() & a2}
d2= pd.DataFrame.from_dict(a3, orient='index')

d5= pd.DataFrame.from_dict({k: v  for k, v in c1.items()}, orient='index')
impact = pd.merge(d2, d1, left_index=True, right_index=True)

impact_check1 = pd.merge(d, d5, left_index=True, right_index=True)
impact_check1.to_csv("impact_check1.csv")

impact1=impact.reset_index()
impact2=impact1.sort_values(by=['index'])

# -- generate plots
# histogram

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True,
                 'legend.fontsize': 10,
          'legend.handlelength': 2})
fig, ax = plt.subplots(1, 1)
ax2 = ax.twinx()

a = range(1, len(impact2['index'])+1)
ind = np.arange(len(a))
width =1

plt1= ax.bar(ind+width+0.25, impact2.iloc[:,2], width=0.5,color= "skyblue", lw=0, alpha=0.5, label = 'total claims')
ax.set_xlabel('Rating')

ax2.set_ylabel('proportion of fatigue claims to total claims (%) ')
plt2= ax2.bar(ind+width+0.75, impact2.iloc[:,1],width=0.5, color= "green", lw=0, alpha=0.5, label = 'proportion of fatigue claims to total claims')
ax.set_ylabel('total claims (%)')
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=1,bbox_to_anchor=(0.85, 1.2))
ax.set_xticks(ind+width+(width/2))
ax.set_xticklabels(impact2['index'].tolist(),rotation=90)

#plt.title("loss severity")
plt.show()

ax.set_xticks(ind+width+(width/2))
ax.set_xticklabels(impact2['index'].tolist(),rotation=90)

plt.title("loss severity")
plt.show()


plt.savefig('impact.png')

impact_fatigue['hour']= impact_fatigue['claim_loss_time'].apply(strip_time)
gb_fatigue=impact_fatigue.groupby(['hour','impact_rating_group']).size().reset_index()
gb_fatigue.columns=['hour', 'impact_rating_group', 'count']
impact_all['hour']= impact_all['claim_loss_time'].apply(strip_time)
gb_all=impact_all.groupby(['hour','impact_rating_group']).size().reset_index()
gb_all.columns=['hour', 'impact_rating_group', 'count']
plot_fatigue = gb_fatigue.pivot(index='impact_rating_group', columns='hour', values='count')
plot_all = gb_all.pivot(index='impact_rating_group', columns='hour', values='count')

plt.clf()
#  heatmap
import seaborn as sns
sns_plot1=sns.heatmap(plot_all, cmap="YlGnBu").set_title('total claims')

fig = sns_plot1.get_figure()
fig.savefig("heatmap_allclaims2.png")
fig.show()

plt.clf()
sns_plot2=sns.heatmap(plot_fatigue,cmap="YlGnBu").set_title('fatigue claims')

fig2= sns_plot2.get_figure()
fig2.show()
fig2.savefig("fatigueclaims2.png")

################################ holiday vs non-holiday #############

holiday=pd.read_csv("public_holidays.csv", header=None)
holiday.columns=['date']

def change_date(x):
    return datetime.fromtimestamp(mktime(time.strptime(str(x), '%Y%m%d'))).date()


def get_weekday(x):
    return pd.Timestamp(x).date().weekday()


def check_holiday(x):
    if x in holiday_f.date.tolist():
        idx = np.where(x == holiday_f.date)[0][0]
        # if holiday_f.iloc[idx, 1]=='Long weekend' and get_weekday(x)==0:
        #     return 'long weekend- Mon'
        # elif holiday_f.iloc[idx, 1]=='Long weekend' and get_weekday(x)==1:
        #     return 'long weekend- Tue'
        # elif holiday_f.iloc[idx, 1]=='Long weekend' and get_weekday(x)==5:
        #     return 'long weekend- Sat'
        # elif holiday_f.iloc[idx, 1]=='Long weekend' and get_weekday(x)==6:
        #     return 'long weekend- Sun'
        # elif holiday_f.iloc[idx, 1]=='Long weekend' and get_weekday(x)==4:
        #     return 'long weekend- Fri'
        # elif holiday_f.iloc[idx, 1]=='Long weekend' and get_weekday(x)==3:
        #     return 'long weekend- Thurs'

        if holiday_f.iloc[idx, 1]=='Long weekend':
            return 'long weekend'

        else:
            return 'Public holidays (exclude long weekend)'
    else:
        if get_weekday(x) == 5:
            return 'Sat'
        elif get_weekday(x) == 6:
            return 'Sun'
        else:
            return 'Non-Public holidays'

def check_holiday1(x):
    if x in holiday.apply(change_date).values:
        # if get_weekday(x) in [0,4]:
        #     return 'long weekend'
        # else:
        #     return \
        return 'Public holidays/weekends'
    else:
        if get_weekday(x) == 5:
            return 'Public holidays/weekends'
        elif get_weekday(x) == 6:
            return 'Public holidays/weekends'
        else:

            return 'Non-Public holidays'

from itertools import chain

def find_weekday(x):
    weekday = get_weekday(x)
    if weekday == 0:
        return 'Mon'
    elif weekday == 1:
        return 'Tue'
    elif weekday == 2:
        return 'Wed'
    elif weekday == 3:
        return 'Thur'
    elif weekday == 4:
        return 'Fri'
    elif weekday == 5:
        return 'Sat'
    else:
        return 'Sun'


def find_longweekend(row):
    lst = []
    lst2=[]
    weekday = get_weekday(row['date1'])
    day = row['date1']
    if weekday == 4: # Fri
        lst1 = 'Long weekend'
        day += timedelta(1)
        if day not in holiday.iloc[:, 2].tolist():
            lst.append(day)
        else:
            lst2.append(holiday.loc[holiday.date1 == day, :].index)
        day += timedelta(1)
        if day not in holiday.iloc[:, 2].tolist():
            lst.append(day)
        else:
            lst2.append(holiday.loc[holiday.date1 == day, :].index)


    elif weekday == 0: # mon
        lst1 = 'Long weekend'
        day -= timedelta(1)
        if day not in holiday.iloc[:, 2].tolist():
            lst.append(day)
        else:
            lst2.append(holiday.loc[holiday.date1 == day, :].index)
        day -= timedelta(1)
        if day not in holiday.iloc[:, 2].tolist():
            lst.append(day)
        else:
            lst2.append(holiday.loc[holiday.date1 == day, :].index)

    else:
        lst1 = row['weekday']

    return lst, lst1, lst2


def find_longweekend1(row):
    lst = []
    weekday = get_weekday(row['date'])
    day = row['date']
    if weekday == 3:  # Thurs
        if day in holiday_f.date.tolist():
            day += timedelta(1)
            if day in holiday_f.date.tolist(): # check if the next day is also a public holiday
                lst.append(row['date'])

    elif weekday == 1: # Tue
        if day in holiday_f.date.tolist():
            day -= timedelta(1)

            if day in holiday_f.date.tolist():  # check if the previous day is also a public holiday
                lst.append(row['date'])
    return lst

impact_all2 = impact_all.copy()
impact_fatigue2= impact_fatigue.copy()
holiday['weekday'] = holiday.iloc[:, 0].apply(change_date).apply(find_weekday)
holiday['date1'] = holiday.iloc[:, 0].apply(change_date)
lst = holiday.apply(find_longweekend, axis=1)
lst2= [k for i, v, k in lst]
lst2_1= pd.DataFrame([i for i in list(chain(*lst2)) if len(i)> 0], columns=['indx']).drop_duplicates()
holiday['weekday1'] = [v for i, v ,k in lst]
holiday4 = holiday[['date1', 'weekday1']]
holiday4.columns = ['date', 'weekday']
holiday4.loc[lst2_1.indx,'weekday'] = 'Long weekend'
holiday2 = [i for i, v, k in lst]
holiday3 = pd.DataFrame(list(chain(*holiday2)), columns=['date'])
holiday3['weekday'] = 'Long weekend'
holiday_f = pd.concat([holiday4, holiday3], axis=0)
holiday_f=holiday_f.reset_index(drop=True)
lst3_1= [ holiday_f.loc[holiday_f.date == x, :].index[0] for x in list(chain(*holiday_f.apply(find_longweekend1, axis=1)))]
holiday_f.loc[lst3_1, 'weekday']='Long weekend'
holiday_f=holiday_f.drop_duplicates()
impact_all2['holiday1'] = impact_all2.claim_loss_date.apply(check_holiday)
impact_fatigue2['holiday1'] = impact_fatigue2.claim_loss_date.apply(check_holiday)

c=Counter(impact_all2.holiday1)
holiday2_count1={k:v /sum(Counter(impact_all2.holiday1).values())*100 for k, v in c.items()}

c1 = Counter(impact_fatigue2.holiday1)
holiday2_count2={k:v /sum(Counter(impact_fatigue2.holiday1).values()) for k, v in c1.items()}

a_2 = {k:v for k, v in c1.items()}
a2_2 = {k: v for k, v in c.items()}

#all _claims
d1= pd.DataFrame.from_dict(holiday2_count1, orient='index')

a3_2= {k: a_2[k]/a2_2[k]*100 for k in a_2.keys() & a2_2}
sum1=sum(Counter(holiday2_count1).values())*100
a4_2= {k: a3_2[k]*holiday2_count1[k] for k in a2_2.keys() & a3_2}
d2= pd.DataFrame.from_dict(a3_2, orient='index')

holiday_data = pd.merge(d2, d1, left_index=True, right_index=True)
test=b5_1.reset_index(drop=True)

holiday_data  =test.reindex([1,0,2,3,4])

# ------plot ----
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True,
                 'legend.fontsize': 10,
               'legend.handlelength': 2})
a = range(1,6)
ind = np.arange(len(a))
width = 1

fig= plt.figure(figsize=(30,10))
ax = fig.add_subplot(111)

ax.bar(ind+width-0.25, holiday_data.iloc[:,1], 0.5, color='#b0c4de', label='total claims')
ax2 = ax.twinx()
ax2.bar(ind+width+0.15,  holiday_data.iloc[:,0], 0.5, color='#deb0b0', label='proportion of fatigue claims to total claims')
rects = ax.patches
ax.set_xticks(ind+width)#+(width/2))
ax.set_xticklabels(['Non Public holiday','Sat','Sun', 'long weekend', 'Public holiday \n(exclude long weekend)'], rotation=45)
labels = ["{} % ".format(str(round(i,2))) for i in b5_1.iloc[:,1]]
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height , label,
            ha='center', va='bottom')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax2.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(15)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax2.legend(lines + lines2, labels + labels2, loc=0, fontsize='x-large')
ax.xaxis.label.set_size(10)

ax.set_ylabel("total claims (%)")
ax2.set_ylabel("proportion of fatigue claims to total claims (%)")
#ax.set_ylim(0,100)
#ax2.set_ylim(0,100)
plt.show()
plt.savefig('holiday.png')

################################  text mining    ###############################
nlp= spacy.load('en')
# text preprocessing
def process_txt(wordList):
    pattern2 = re.compile(r'UNKNOWN REGO|REGO UNKNOWN', re.DOTALL|re.IGNORECASE)   # replace UNKNOWN REGO with word 'vehicle'
    wordList = [re.sub(pattern2, ' vehihle ', word) for word in wordList]
    pattern = re.compile(r'[0-9A-Z]{6}')                                           # replace rego number with word 'vehicle'
    wordList= [re.sub(pattern, 'vehicle', word) for word in wordList]
    wordList = [re.sub('[^0-9a-zA-Z]+', ' ', word).lower() for word in wordList]   # strip all punctuation marks and change to lowercase
    pattern1 = re.compile(r'\scar\s|car\.|car\s')
    wordList=[re.sub(pattern1, ' vehicle ', word) for word in wordList]             # replace word 'car' with word 'vehicle'
    wordList= [re.sub(pattern, 'vehicle', word) for word in wordList]
    return wordList

stops=stopwords.words("english")

doc1 = [nlp(doc) for doc in process_txt(df_extracted.claim_description)]
#and token.pos_ not in ['ADJ', 'ADV']

warnings.filterwarnings('ignore')

docs = [[token.lemma_ for token in doc if token.text not in stops] for doc in doc1] #-- remove stopwords and lemmatize the text

bigram = Phrases(docs, min_count=300)

#-- add bigrams to the corpus --
bigrams = []
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            docs[idx].append(token)
            bigrams.append(token)


# -- generate  word cloud
word_cloud_text= [' '.join([i for i in doc]) for doc in docs]

tfidf_vectorizer = TfidfVectorizer(max_df=0.7, min_df=50,  # calculate tfidf weight
                                   max_features=20000,
                                   stop_words='english', ngram_range=(1, 1))

tfidf = tfidf_vectorizer.fit_transform(word_cloud_text)

freqs = [(word,  tfidf.getcol(idx).sum()) for word, idx in tfidf_vectorizer.vocabulary_.items()]

d= {}
for k, v in freqs:
    d[k]=v

wc = WordCloud(
    background_color="white",
    max_words=2000,
    width = 1024,
    height = 720,
    stopwords=stopwords.words("english")
)

wc.generate_from_frequencies(d)
plt.imshow(wc, interpolation='bilinear')
plt.title("word cloud")
plt.axis("off")
plt.show()
plt.savefig('wordcloud1.png')

# -- run LDA on text

dictionary = Dictionary(docs)
dictionary.filter_extremes(no_below=20, no_above=0.65, keep_n=10000)
corpus = [dictionary.doc2bow(text) for text in docs]

#tfidf = TfidfModel(corpus)
#corpus_tfidf = tfidf[corpus]
n_topics = 3
lda = LdaModel(corpus, id2word=dictionary, num_topics=n_topics)
ldatopics = lda.show_topics(formatted=False)

pyLDAvis.enable_notebook()
p= pyLDAvis.gensim.prepare(lda, corpus, dictionary, R=6)
pyLDAvis.save_html(p, 'lda_topic_modeling1.html')


##################################



