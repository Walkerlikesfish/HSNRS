import lmdb

db_root = '../VH/'

env = lmdb.open("")
txn = env.begin(write=True)

database1 = txn.cursor("db1Name")
database2 = txn.cursor("db2Name")

env.open_db(key="newDBName", txn=txn)
newDatabase = txt.cursor("newDBName")

for (key, value) in database1:
    newDatabase.put(key, value)

for (key, value) in database2:
    newDatabase.put(key, value)