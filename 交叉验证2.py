
1. Libraries and Data
2. Catboost GPU
3. Features importance
4. Submission
1. Librairies and data
import numpy as np 
import pandas as pd 
# Data processing, metrics and modeling
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from bayes_opt import BayesianOptimization
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve

from sklearn import metrics
from sklearn import preprocessing
import catboost
from catboost import Pool

# Suppr warning
import warnings
warnings.filterwarnings("ignore")

import itertools
from scipy import interp

# Plots
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import rcParams

#Timer
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken for Modeling: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
DATASETS
%%time
train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')
train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')
sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')
CPU times: user 40.2 s, sys: 3.97 s, total: 44.1 s
Wall time: 44.3 s
MERGE, MISSING VALUE, FILL NA
# merge 
train_df = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test_df = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print("Train shape : "+str(train_df.shape))
print("Test shape  : "+str(test_df.shape))
Train shape : (590540, 433)
Test shape  : (506691, 432)
pd.set_option('display.max_columns', 500)
# GPreda, missing data
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))
display(missing_data(train_df), missing_data(test_df))
isFraud	TransactionDT	TransactionAmt	ProductCD	card1	card2	card3	card4	card5	card6	addr1	addr2	dist1	dist2	P_emaildomain	R_emaildomain	C1	C2	C3	C4	C5	C6	C7	C8	C9	C10	C11	C12	C13	C14	D1	D2	D3	D4	D5	D6	D7	D8	D9	D10	D11	D12	D13	D14	D15	M1	M2	M3	M4	M5	M6	M7	M8	M9	V1	V2	V3	V4	V5	V6	V7	V8	V9	V10	V11	V12	V13	V14	V15	V16	V17	V18	V19	V20	V21	V22	V23	V24	V25	V26	V27	V28	V29	V30	V31	V32	V33	V34	V35	V36	V37	V38	V39	V40	V41	V42	V43	V44	V45	V46	V47	V48	V49	V50	V51	V52	V53	V54	V55	V56	V57	V58	V59	V60	V61	V62	V63	V64	V65	V66	V67	V68	V69	V70	V71	V72	V73	V74	V75	V76	V77	V78	V79	V80	V81	V82	V83	V84	V85	V86	V87	V88	V89	V90	V91	V92	V93	V94	V95	V96	V97	V98	V99	V100	V101	V102	V103	V104	V105	V106	V107	V108	V109	V110	V111	V112	V113	V114	V115	V116	V117	V118	V119	V120	V121	V122	V123	V124	V125	V126	V127	V128	V129	V130	V131	V132	V133	V134	V135	V136	V137	V138	V139	V140	V141	V142	V143	V144	V145	V146	V147	V148	V149	V150	V151	V152	V153	V154	V155	V156	V157	V158	V159	V160	V161	V162	V163	V164	V165	V166	V167	V168	V169	V170	V171	V172	V173	V174	V175	V176	V177	V178	V179	V180	V181	V182	V183	V184	V185	V186	V187	V188	V189	V190	V191	V192	V193	V194	V195	V196	V197	V198	V199	V200	V201	V202	V203	V204	V205	V206	V207	V208	V209	V210	V211	V212	V213	V214	V215	V216	V217	V218	V219	V220	V221	V222	V223	V224	V225	V226	V227	V228	V229	V230	V231	V232	V233	V234	V235	V236	V237	V238	V239	V240	V241	V242	V243	V244	V245	V246	V247	V248	V249	V250	V251	V252	V253	V254	V255	V256	V257	V258	V259	V260	V261	V262	V263	V264	V265	V266	V267	V268	V269	V270	V271	V272	V273	V274	V275	V276	V277	V278	V279	V280	V281	V282	V283	V284	V285	V286	V287	V288	V289	V290	V291	V292	V293	V294	V295	V296	V297	V298	V299	V300	V301	V302	V303	V304	V305	V306	V307	V308	V309	V310	V311	V312	V313	V314	V315	V316	V317	V318	V319	V320	V321	V322	V323	V324	V325	V326	V327	V328	V329	V330	V331	V332	V333	V334	V335	V336	V337	V338	V339	id_01	id_02	id_03	id_04	id_05	id_06	id_07	id_08	id_09	id_10	id_11	id_12	id_13	id_14	id_15	id_16	id_17	id_18	id_19	id_20	id_21	id_22	id_23	id_24	id_25	id_26	id_27	id_28	id_29	id_30	id_31	id_32	id_33	id_34	id_35	id_36	id_37	id_38	DeviceType	DeviceInfo
Total	0	0	0	0	0	8933	1565	1577	4259	1571	65706	65706	352271	552913	94456	453249	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1269	280797	262878	168922	309841	517353	551623	515614	515614	76022	279287	525823	528588	528353	89113	271100	271100	271100	281444	350482	169360	346265	346252	346252	279287	279287	279287	279287	279287	279287	279287	279287	279287	279287	279287	76073	76073	76073	76073	76073	76073	76073	76073	76073	76073	76073	76073	76073	76073	76073	76073	76073	76073	76073	76073	76073	76073	76073	168969	168969	168969	168969	168969	168969	168969	168969	168969	168969	168969	168969	168969	168969	168969	168969	168969	168969	77096	77096	77096	77096	77096	77096	77096	77096	77096	77096	77096	77096	77096	77096	77096	77096	77096	77096	77096	77096	77096	77096	89164	89164	89164	89164	89164	89164	89164	89164	89164	89164	89164	89164	89164	89164	89164	89164	89164	89164	89164	89164	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	314	508595	508595	508595	508595	508595	508589	508589	508589	508595	508595	508595	508595	508589	508589	508589	508595	508595	508595	508595	508595	508595	508589	508589	508595	508595	508595	508589	508589	508589	450909	450909	450721	450721	450721	450909	450909	450721	450721	450909	450909	450909	450909	450721	450909	450909	450909	450721	450721	450909	450909	450721	450721	450909	450909	450909	450909	450721	450721	450909	450721	450721	450909	450721	450721	450909	450909	450909	450909	450909	450909	450721	450721	450721	450909	450909	450909	450909	450909	450909	460110	460110	460110	449124	449124	449124	460110	460110	460110	460110	449124	460110	460110	460110	460110	460110	460110	449124	460110	460110	460110	449124	449124	460110	460110	460110	460110	460110	449124	460110	460110	460110	460110	449124	449124	460110	460110	460110	449124	449124	460110	460110	449124	460110	460110	460110	460110	460110	460110	460110	460110	460110	460110	449124	449124	449124	460110	460110	460110	460110	460110	460110	12	12	1269	1269	1269	12	12	12	12	1269	1269	12	12	12	12	12	12	1269	12	12	12	1269	1269	12	12	12	12	12	12	12	12	12	12	12	1269	1269	1269	12	12	12	12	12	12	508189	508189	508189	508189	508189	508189	508189	508189	508189	508189	508189	508189	508189	508189	508189	508189	508189	508189	446307	449668	524216	524216	453675	453675	585385	585385	515614	515614	449562	446307	463220	510496	449555	461200	451171	545427	451222	451279	585381	585371	585371	585793	585408	585377	585371	449562	449562	512975	450258	512954	517251	512735	449555	449555	449555	449555	449730	471874
Percent	0	0	0	0	0	1.51268	0.265012	0.267044	0.721204	0.266028	11.1264	11.1264	59.6524	93.6284	15.9949	76.7516	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.214888	47.5492	44.5149	28.6047	52.4674	87.6068	93.4099	87.3123	87.3123	12.8733	47.2935	89.041	89.5093	89.4695	15.0901	45.9071	45.9071	45.9071	47.6588	59.3494	28.6788	58.6353	58.6331	58.6331	47.2935	47.2935	47.2935	47.2935	47.2935	47.2935	47.2935	47.2935	47.2935	47.2935	47.2935	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	12.8819	28.6126	28.6126	28.6126	28.6126	28.6126	28.6126	28.6126	28.6126	28.6126	28.6126	28.6126	28.6126	28.6126	28.6126	28.6126	28.6126	28.6126	28.6126	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	13.0552	15.0987	15.0987	15.0987	15.0987	15.0987	15.0987	15.0987	15.0987	15.0987	15.0987	15.0987	15.0987	15.0987	15.0987	15.0987	15.0987	15.0987	15.0987	15.0987	15.0987	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	0.0531717	86.1237	86.1237	86.1237	86.1237	86.1237	86.1227	86.1227	86.1227	86.1237	86.1237	86.1237	86.1237	86.1227	86.1227	86.1227	86.1237	86.1237	86.1237	86.1237	86.1237	86.1237	86.1227	86.1227	86.1237	86.1237	86.1237	86.1227	86.1227	86.1227	76.3554	76.3554	76.3235	76.3235	76.3235	76.3554	76.3554	76.3235	76.3235	76.3554	76.3554	76.3554	76.3554	76.3235	76.3554	76.3554	76.3554	76.3235	76.3235	76.3554	76.3554	76.3235	76.3235	76.3554	76.3554	76.3554	76.3554	76.3235	76.3235	76.3554	76.3235	76.3235	76.3554	76.3235	76.3235	76.3554	76.3554	76.3554	76.3554	76.3554	76.3554	76.3235	76.3235	76.3235	76.3554	76.3554	76.3554	76.3554	76.3554	76.3554	77.9134	77.9134	77.9134	76.0531	76.0531	76.0531	77.9134	77.9134	77.9134	77.9134	76.0531	77.9134	77.9134	77.9134	77.9134	77.9134	77.9134	76.0531	77.9134	77.9134	77.9134	76.0531	76.0531	77.9134	77.9134	77.9134	77.9134	77.9134	76.0531	77.9134	77.9134	77.9134	77.9134	76.0531	76.0531	77.9134	77.9134	77.9134	76.0531	76.0531	77.9134	77.9134	76.0531	77.9134	77.9134	77.9134	77.9134	77.9134	77.9134	77.9134	77.9134	77.9134	77.9134	76.0531	76.0531	76.0531	77.9134	77.9134	77.9134	77.9134	77.9134	77.9134	0.00203204	0.00203204	0.214888	0.214888	0.214888	0.00203204	0.00203204	0.00203204	0.00203204	0.214888	0.214888	0.00203204	0.00203204	0.00203204	0.00203204	0.00203204	0.00203204	0.214888	0.00203204	0.00203204	0.00203204	0.214888	0.214888	0.00203204	0.00203204	0.00203204	0.00203204	0.00203204	0.00203204	0.00203204	0.00203204	0.00203204	0.00203204	0.00203204	0.214888	0.214888	0.214888	0.00203204	0.00203204	0.00203204	0.00203204	0.00203204	0.00203204	86.055	86.055	86.055	86.055	86.055	86.055	86.055	86.055	86.055	86.055	86.055	86.055	86.055	86.055	86.055	86.055	86.055	86.055	75.5761	76.1452	88.7689	88.7689	76.8238	76.8238	99.1271	99.1271	87.3123	87.3123	76.1273	75.5761	78.4401	86.4456	76.1261	78.098	76.3997	92.3607	76.4084	76.418	99.1264	99.1247	99.1247	99.1962	99.131	99.1257	99.1247	76.1273	76.1273	86.8654	76.2451	86.8619	87.5895	86.8248	76.1261	76.1261	76.1261	76.1261	76.1557	79.9055
Types	int64	int64	float64	object	int64	float64	float64	object	float64	object	float64	float64	float64	float64	object	object	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	object	object	object	object	object	object	object	object	object	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	object	float64	float64	object	object	float64	float64	float64	float64	float64	float64	object	float64	float64	float64	object	object	object	object	object	float64	object	object	object	object	object	object	object	object
TransactionDT	TransactionAmt	ProductCD	card1	card2	card3	card4	card5	card6	addr1	addr2	dist1	dist2	P_emaildomain	R_emaildomain	C1	C2	C3	C4	C5	C6	C7	C8	C9	C10	C11	C12	C13	C14	D1	D2	D3	D4	D5	D6	D7	D8	D9	D10	D11	D12	D13	D14	D15	M1	M2	M3	M4	M5	M6	M7	M8	M9	V1	V2	V3	V4	V5	V6	V7	V8	V9	V10	V11	V12	V13	V14	V15	V16	V17	V18	V19	V20	V21	V22	V23	V24	V25	V26	V27	V28	V29	V30	V31	V32	V33	V34	V35	V36	V37	V38	V39	V40	V41	V42	V43	V44	V45	V46	V47	V48	V49	V50	V51	V52	V53	V54	V55	V56	V57	V58	V59	V60	V61	V62	V63	V64	V65	V66	V67	V68	V69	V70	V71	V72	V73	V74	V75	V76	V77	V78	V79	V80	V81	V82	V83	V84	V85	V86	V87	V88	V89	V90	V91	V92	V93	V94	V95	V96	V97	V98	V99	V100	V101	V102	V103	V104	V105	V106	V107	V108	V109	V110	V111	V112	V113	V114	V115	V116	V117	V118	V119	V120	V121	V122	V123	V124	V125	V126	V127	V128	V129	V130	V131	V132	V133	V134	V135	V136	V137	V138	V139	V140	V141	V142	V143	V144	V145	V146	V147	V148	V149	V150	V151	V152	V153	V154	V155	V156	V157	V158	V159	V160	V161	V162	V163	V164	V165	V166	V167	V168	V169	V170	V171	V172	V173	V174	V175	V176	V177	V178	V179	V180	V181	V182	V183	V184	V185	V186	V187	V188	V189	V190	V191	V192	V193	V194	V195	V196	V197	V198	V199	V200	V201	V202	V203	V204	V205	V206	V207	V208	V209	V210	V211	V212	V213	V214	V215	V216	V217	V218	V219	V220	V221	V222	V223	V224	V225	V226	V227	V228	V229	V230	V231	V232	V233	V234	V235	V236	V237	V238	V239	V240	V241	V242	V243	V244	V245	V246	V247	V248	V249	V250	V251	V252	V253	V254	V255	V256	V257	V258	V259	V260	V261	V262	V263	V264	V265	V266	V267	V268	V269	V270	V271	V272	V273	V274	V275	V276	V277	V278	V279	V280	V281	V282	V283	V284	V285	V286	V287	V288	V289	V290	V291	V292	V293	V294	V295	V296	V297	V298	V299	V300	V301	V302	V303	V304	V305	V306	V307	V308	V309	V310	V311	V312	V313	V314	V315	V316	V317	V318	V319	V320	V321	V322	V323	V324	V325	V326	V327	V328	V329	V330	V331	V332	V333	V334	V335	V336	V337	V338	V339	id_01	id_02	id_03	id_04	id_05	id_06	id_07	id_08	id_09	id_10	id_11	id_12	id_13	id_14	id_15	id_16	id_17	id_18	id_19	id_20	id_21	id_22	id_23	id_24	id_25	id_26	id_27	id_28	id_29	id_30	id_31	id_32	id_33	id_34	id_35	id_36	id_37	id_38	DeviceType	DeviceInfo
Total	0	0	0	0	8654	3002	3086	4547	3007	65609	65609	291217	470255	69192	370821	3	3	3	3	3	3	3	3	3	3	3	3	4748	3	6031	234769	203142	76851	224375	381908	446558	432353	432353	12545	176518	437437	383307	391497	12069	176639	176639	176639	237745	309632	158939	235018	235004	235004	176518	176518	176518	176518	176518	176518	176518	176518	176518	176518	176518	12589	12589	12589	12589	12589	12589	12589	12589	12589	12589	12589	12589	12589	12589	12589	12589	12589	12589	12589	12589	12589	12589	12589	76854	76854	76854	76854	76854	76854	76854	76854	76854	76854	76854	76854	76854	76854	76854	76854	76854	76854	12899	12899	12899	12899	12899	12899	12899	12899	12899	12899	12899	12899	12899	12899	12899	12899	12899	12899	12899	12899	12899	12899	12081	12081	12081	12081	12081	12081	12081	12081	12081	12081	12081	12081	12081	12081	12081	12081	12081	12081	12081	12081	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	430906	430906	430906	430906	430906	430636	430636	430636	430906	430906	430906	430906	430636	430636	430636	430906	430906	430906	430906	430906	430906	430636	430636	430906	430906	430906	430636	430636	430636	369957	369957	370316	370316	370316	369957	369957	370316	370316	369957	369957	369957	369957	370316	369957	369957	369957	370316	370316	369957	369957	370316	370316	369957	369957	369957	369957	370316	370316	369957	370316	370316	369957	370316	370316	369957	369957	369957	369957	369957	369957	370316	370316	370316	369957	369957	369957	369957	369957	369957	379963	379963	379963	369375	369375	369375	379963	379963	379963	379963	369375	379963	379963	379963	379963	379963	379963	369375	379963	379963	379963	369375	369375	379963	379963	379963	379963	379963	369375	379963	379963	379963	379963	369375	369375	379963	379963	379963	369375	369375	379963	379963	369375	379963	379963	379963	379963	379963	379963	379963	379963	379963	379963	369375	369375	369375	379963	379963	379963	379963	379963	379963	3	3	6031	6031	6031	3	3	3	3	6031	6031	3	3	3	3	3	3	6031	3	3	3	6031	6031	3	3	3	3	3	3	3	3	3	3	3	6031	6031	6031	3	3	3	3	3	3	430260	430260	430260	430260	430260	430260	430260	430260	430260	430260	430260	430260	430260	430260	430260	430260	430260	430260	364784	369715	440210	440210	371941	371941	501632	501632	432353	432353	369913	364784	376405	435334	369714	380944	370725	455816	370785	371058	501632	501629	501629	501951	501652	501644	501629	369913	369913	436032	370066	436020	436020	434516	369714	369714	369714	369714	369760	391634
Percent	0	0	0	0	1.70794	0.592472	0.60905	0.897391	0.593458	12.9485	12.9485	57.4743	92.809	13.6557	73.1848	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	0.93706	0.000592077	1.19027	46.3338	40.0919	15.1672	44.2824	75.373	88.1322	85.3287	85.3287	2.47587	34.8374	86.3321	75.6491	77.2654	2.38193	34.8613	34.8613	34.8613	46.9211	61.1086	31.368	46.3829	46.3801	46.3801	34.8374	34.8374	34.8374	34.8374	34.8374	34.8374	34.8374	34.8374	34.8374	34.8374	34.8374	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	2.48455	15.1678	15.1678	15.1678	15.1678	15.1678	15.1678	15.1678	15.1678	15.1678	15.1678	15.1678	15.1678	15.1678	15.1678	15.1678	15.1678	15.1678	15.1678	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.54573	2.38429	2.38429	2.38429	2.38429	2.38429	2.38429	2.38429	2.38429	2.38429	2.38429	2.38429	2.38429	2.38429	2.38429	2.38429	2.38429	2.38429	2.38429	2.38429	2.38429	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	85.0432	85.0432	85.0432	85.0432	85.0432	84.9899	84.9899	84.9899	85.0432	85.0432	85.0432	85.0432	84.9899	84.9899	84.9899	85.0432	85.0432	85.0432	85.0432	85.0432	85.0432	84.9899	84.9899	85.0432	85.0432	85.0432	84.9899	84.9899	84.9899	73.0143	73.0143	73.0852	73.0852	73.0852	73.0143	73.0143	73.0852	73.0852	73.0143	73.0143	73.0143	73.0143	73.0852	73.0143	73.0143	73.0143	73.0852	73.0852	73.0143	73.0143	73.0852	73.0852	73.0143	73.0143	73.0143	73.0143	73.0852	73.0852	73.0143	73.0852	73.0852	73.0143	73.0852	73.0852	73.0143	73.0143	73.0143	73.0143	73.0143	73.0143	73.0852	73.0852	73.0852	73.0143	73.0143	73.0143	73.0143	73.0143	73.0143	74.9891	74.9891	74.9891	72.8995	72.8995	72.8995	74.9891	74.9891	74.9891	74.9891	72.8995	74.9891	74.9891	74.9891	74.9891	74.9891	74.9891	72.8995	74.9891	74.9891	74.9891	72.8995	72.8995	74.9891	74.9891	74.9891	74.9891	74.9891	72.8995	74.9891	74.9891	74.9891	74.9891	72.8995	72.8995	74.9891	74.9891	74.9891	72.8995	72.8995	74.9891	74.9891	72.8995	74.9891	74.9891	74.9891	74.9891	74.9891	74.9891	74.9891	74.9891	74.9891	74.9891	72.8995	72.8995	72.8995	74.9891	74.9891	74.9891	74.9891	74.9891	74.9891	0.000592077	0.000592077	1.19027	1.19027	1.19027	0.000592077	0.000592077	0.000592077	0.000592077	1.19027	1.19027	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	1.19027	0.000592077	0.000592077	0.000592077	1.19027	1.19027	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	1.19027	1.19027	1.19027	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	0.000592077	84.9157	84.9157	84.9157	84.9157	84.9157	84.9157	84.9157	84.9157	84.9157	84.9157	84.9157	84.9157	84.9157	84.9157	84.9157	84.9157	84.9157	84.9157	71.9934	72.9666	86.8794	86.8794	73.4059	73.4059	99.0016	99.0016	85.3287	85.3287	73.0056	71.9934	74.2869	85.9171	72.9664	75.1827	73.1659	89.9594	73.1777	73.2316	99.0016	99.001	99.001	99.0645	99.0055	99.0039	99.001	73.0056	73.0056	86.0548	73.0358	86.0524	86.0524	85.7556	72.9664	72.9664	72.9664	72.9664	72.9754	77.2925
Types	int64	float64	object	int64	float64	float64	object	float64	object	float64	float64	float64	float64	object	object	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	object	object	object	object	object	object	object	object	object	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	float64	object	float64	float64	object	object	float64	float64	float64	float64	float64	float64	object	float64	float64	float64	object	object	object	object	object	float64	object	object	object	object	object	object	object	object
#fillna
train_df = train_df.fillna(-999)
test_df = test_df.fillna(-999)
del train_transaction, train_identity, test_transaction, test_identity
ENCODING
# Label Encoding
for f in train_df.columns:
    if  train_df[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values) + list(test_df[f].values))
        train_df[f] = lbl.transform(list(train_df[f].values))
        test_df[f] = lbl.transform(list(test_df[f].values))  
train_df = train_df.reset_index()
test_df = test_df.reset_index()
features = list(train_df)
features.remove('isFraud')
target = 'isFraud'
CONFUSION MATRIX
2. Catboost
PARAMS
param_cb = {
        'learning_rate': 0.2,
        'bagging_temperature': 0.1, 
        'l2_leaf_reg': 30,
        'depth': 12, 
        'max_leaves': 48,
        'max_bin':255,
        'iterations' : 1000,
        'task_type':'GPU',
        'loss_function' : "Logloss",
        'objective':'CrossEntropy',
        'eval_metric' : "AUC",
        'bootstrap_type' : 'Bayesian',
        'random_seed':1337,
        'early_stopping_rounds' : 100,
        'use_best_model': True 
}
CV 5 FOLDS AND METRICS
print('CatBoost GPU modeling...')
start_time = timer(None)
plt.rcParams["axes.grid"] = True

nfold = 5
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)

oof = np.zeros(len(train_df))
mean_fpr = np.linspace(0,1,100)
cms= []
tprs = []
aucs = []
y_real = []
y_proba = []
recalls = []
roc_aucs = []
f1_scores = []
accuracies = []
precisions = []
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

i = 1
for train_idx, valid_idx in skf.split(train_df, train_df.isFraud.values):
    print("\nfold {}".format(i))
    trn_data = Pool(train_df.iloc[train_idx][features].values,
                   label=train_df.iloc[train_idx][target].values
                   )
    val_data = Pool(train_df.iloc[valid_idx][features].values,
                   label=train_df.iloc[valid_idx][target].values
                   )   

    clf = catboost.train(trn_data, param_cb, eval_set= val_data, verbose = 300)

    oof[valid_idx]  = clf.predict(train_df.iloc[valid_idx][features].values)   
    oof[valid_idx]  = np.exp(oof[valid_idx]) / (1 + np.exp(oof[valid_idx]))
    
    predictions += clf.predict(test_df[features]) / nfold
    predictions = np.exp(predictions)/(1 + np.exp(predictions))
    
    # Scores 
    roc_aucs.append(roc_auc_score(train_df.iloc[valid_idx][target].values, oof[valid_idx]))
    accuracies.append(accuracy_score(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))
    recalls.append(recall_score(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))
    precisions.append(precision_score(train_df.iloc[valid_idx][target].values ,oof[valid_idx].round()))
    f1_scores.append(f1_score(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))
    
    # Roc curve by fold
    f = plt.figure(1)
    fpr, tpr, t = roc_curve(train_df.iloc[valid_idx][target].values, oof[valid_idx])
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i,roc_auc))

    # Precion recall by folds
    g = plt.figure(2)
    precision, recall, _ = precision_recall_curve(train_df.iloc[valid_idx][target].values, oof[valid_idx])
    y_real.append(train_df.iloc[valid_idx][target].values)
    y_proba.append(oof[valid_idx])
    plt.plot(recall, precision, lw=2, alpha=0.3, label='P|R fold %d' % (i))  
    
    i= i+1
    
    # Confusion matrix by folds
    cms.append(confusion_matrix(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))
    
    # Features imp
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.get_feature_importance()
    fold_importance_df["fold"] = nfold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

# Metrics
print(
        '\nCV roc score        : {0:.4f}, std: {1:.4f}.'.format(np.mean(roc_aucs), np.std(roc_aucs)),
        '\nCV accuracy score   : {0:.4f}, std: {1:.4f}.'.format(np.mean(accuracies), np.std(accuracies)),
        '\nCV recall score     : {0:.4f}, std: {1:.4f}.'.format(np.mean(recalls), np.std(recalls)),
        '\nCV precision score  : {0:.4f}, std: {1:.4f}.'.format(np.mean(precisions), np.std(precisions)),
        '\nCV f1 score         : {0:.4f}, std: {1:.4f}.'.format(np.mean(f1_scores), np.std(f1_scores))
)

#ROC
f = plt.figure(1)
plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'grey')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.4f)' % (np.mean(roc_aucs)),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Catboost ROC curve by folds')
plt.legend(loc="lower right")

# PR plt
g = plt.figure(2)
plt.plot([0,1],[1,0],linestyle = '--',lw = 2,color = 'grey')
y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba)
plt.plot(recall, precision, color='blue',
         label=r'Mean P|R')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Catboost P|R curve by folds')
plt.legend(loc="lower left")

# Confusion maxtrix & metrics
plt.rcParams["axes.grid"] = False
cm = np.average(cms, axis=0)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cm, 
                      classes=class_names, 
                      title= 'CatBoost Confusion matrix [averaged/folds]')
# Timer end    
timer(start_time)
CatBoost GPU modeling...

fold 1
0:	learn: 0.7910843	test: 0.7868275	best: 0.7868275 (0)	total: 182ms	remaining: 3m 1s
300:	learn: 0.9725139	test: 0.9474699	best: 0.9474699 (300)	total: 48s	remaining: 1m 51s
600:	learn: 0.9860049	test: 0.9570317	best: 0.9570317 (600)	total: 1m 35s	remaining: 1m 3s
900:	learn: 0.9923344	test: 0.9621251	best: 0.9621251 (900)	total: 2m 22s	remaining: 15.6s
999:	learn: 0.9936540	test: 0.9631413	best: 0.9631603 (989)	total: 2m 38s	remaining: 0us
bestTest = 0.963160336
bestIteration = 989
Shrink model to first 990 iterations.

fold 2
0:	learn: 0.7899898	test: 0.7911923	best: 0.7911923 (0)	total: 169ms	remaining: 2m 49s
300:	learn: 0.9698937	test: 0.9499920	best: 0.9499920 (300)	total: 47.6s	remaining: 1m 50s
600:	learn: 0.9844199	test: 0.9600127	best: 0.9600127 (600)	total: 1m 33s	remaining: 1m 2s
900:	learn: 0.9917955	test: 0.9648936	best: 0.9649113 (898)	total: 2m 20s	remaining: 15.4s
999:	learn: 0.9934473	test: 0.9655753	best: 0.9655907 (995)	total: 2m 35s	remaining: 0us
bestTest = 0.9655907154
bestIteration = 995
Shrink model to first 996 iterations.

fold 3
0:	learn: 0.7896961	test: 0.7926685	best: 0.7926685 (0)	total: 171ms	remaining: 2m 51s
300:	learn: 0.9711481	test: 0.9494763	best: 0.9494763 (300)	total: 47.7s	remaining: 1m 50s
600:	learn: 0.9861364	test: 0.9594110	best: 0.9594110 (600)	total: 1m 34s	remaining: 1m 3s
900:	learn: 0.9929530	test: 0.9637629	best: 0.9637629 (900)	total: 2m 22s	remaining: 15.6s
999:	learn: 0.9941906	test: 0.9646859	best: 0.9646874 (994)	total: 2m 37s	remaining: 0us
bestTest = 0.964687407
bestIteration = 994
Shrink model to first 995 iterations.

fold 4
0:	learn: 0.7786031	test: 0.7723401	best: 0.7723401 (0)	total: 189ms	remaining: 3m 9s
300:	learn: 0.9739292	test: 0.9491835	best: 0.9491835 (300)	total: 47.9s	remaining: 1m 51s
600:	learn: 0.9874730	test: 0.9590637	best: 0.9590637 (600)	total: 1m 34s	remaining: 1m 2s
900:	learn: 0.9936475	test: 0.9636875	best: 0.9636875 (900)	total: 2m 21s	remaining: 15.6s
999:	learn: 0.9945901	test: 0.9646183	best: 0.9646380 (997)	total: 2m 37s	remaining: 0us
bestTest = 0.9646379948
bestIteration = 997
Shrink model to first 998 iterations.

fold 5
0:	learn: 0.7909822	test: 0.7886975	best: 0.7886975 (0)	total: 173ms	remaining: 2m 52s
300:	learn: 0.9701827	test: 0.9485023	best: 0.9485041 (299)	total: 47.8s	remaining: 1m 50s
600:	learn: 0.9856985	test: 0.9602432	best: 0.9602432 (600)	total: 1m 35s	remaining: 1m 3s
900:	learn: 0.9929242	test: 0.9655606	best: 0.9655606 (900)	total: 2m 22s	remaining: 15.7s
999:	learn: 0.9942610	test: 0.9670648	best: 0.9670648 (999)	total: 2m 38s	remaining: 0us
bestTest = 0.9670647979
bestIteration = 999
Shrink model to first 1000 iterations.

CV roc score        : 0.9650, std: 0.0013. 
CV accuracy score   : 0.9859, std: 0.0002. 
CV recall score     : 0.6235, std: 0.0054. 
CV precision score  : 0.9589, std: 0.0020. 
CV f1 score         : 0.7557, std: 0.0042.
Time taken for Modeling: 0 hours 21 minutes and 57.79 seconds.



3. Feature importance
plt.style.use('dark_background')
cols = (feature_importance_df[["Feature", "importance"]]
    .groupby("Feature")
    .mean()
    .sort_values(by="importance", ascending=False)[:30].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(10,10))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False),
        edgecolor=('white'), linewidth=2, palette="rocket")
plt.title('CatBoost Features importance (averaged/folds)', fontsize=18)
plt.tight_layout()

4. Submission
sample_submission['isFraud'] = predictions
sample_submission.to_csv('submission_IEEE.csv')
