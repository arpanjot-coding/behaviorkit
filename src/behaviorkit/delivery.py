"""Durable facts + leased at-least-once messages, using standard-library SQLite."""
from __future__ import annotations
from contextlib import closing
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
import time
import uuid

from .contracts import fingerprint
from .core import number, text


@dataclass(frozen=True)
class NotificationPolicy:
    id: str = 'default'
    cooldown_seconds: float = 0
    utc_minutes: tuple = ()
    required_fact: str | None = None

    def __post_init__(self):
        text(self.id,'policy id'); number(self.cooldown_seconds,'cooldown_seconds',0,86400)
        minutes=tuple(tuple(x) for x in self.utc_minutes)
        for interval in minutes:
            if len(interval)!=2 or any(type(v) is not int for v in interval) or not 0<=interval[0]<interval[1]<=1440:
                raise ValueError('UTC schedule requires [start,end) minute pairs in 0..1440')
        object.__setattr__(self,'utc_minutes',minutes)
        if self.required_fact is not None: text(self.required_fact,'required_fact')

    @property
    def fingerprint(self): return fingerprint(asdict(self))

    def eligible(self, *, observed_utc, external_facts, now):
        if self.utc_minutes:
            if observed_utc is None: return False,'unknown_wall_clock'
            number(observed_utc,'observed_utc',0)
            minute=int(observed_utc//60)%1440
            if not any(a<=minute<b for a,b in self.utc_minutes): return False,'outside_schedule'
        if self.required_fact:
            fact=external_facts.get(self.required_fact)
            if not fact: return False,'unknown_external_fact'
            a=number(fact['observed_utc'],'external observed_utc',0)
            b=number(fact['expires_utc'],'external expires_utc',0)
            if b<=a or not a<=now<b: return False,'expired_or_future_external_fact'
            if type(fact['value']) is not bool: raise ValueError('external fact value must be boolean')
            if not fact['value']: return False,'external_fact_false'
        return True,'eligible'


class Outbox:
    def __init__(self, path, *, max_pending=10000, max_facts=100000):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        for value in (max_pending,max_facts):
            if type(value) is not int or value<1: raise ValueError('outbox limits must be positive integers')
        if self.path.is_symlink(): raise ValueError('outbox cannot be a symlink')
        self.max_pending,self.max_facts=max_pending,max_facts
        with closing(self._connect()) as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS facts(id TEXT PRIMARY KEY,payload TEXT NOT NULL,created REAL NOT NULL);
              CREATE TABLE IF NOT EXISTS decisions(id TEXT PRIMARY KEY,event_id TEXT NOT NULL,policy TEXT NOT NULL,destination TEXT NOT NULL,reason TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY,event_id TEXT NOT NULL,destination TEXT NOT NULL,payload TEXT NOT NULL,
                available REAL NOT NULL,lease_until REAL,token TEXT,attempts INTEGER NOT NULL DEFAULT 0,sent REAL,last_error TEXT);
              CREATE INDEX IF NOT EXISTS pending_messages ON messages(sent,available,lease_until);
              CREATE TABLE IF NOT EXISTS cooldowns(key TEXT PRIMARY KEY,accepted REAL NOT NULL);
            ''')
            db.commit()

    def _connect(self):
        if self.path.is_symlink(): raise ValueError('outbox changed to symlink')
        db=sqlite3.connect(self.path,timeout=5,isolation_level=None)
        db.row_factory=sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        return db

    def enqueue(self, event, *, destination='jsonl', policy=None, observed_utc=None, external_facts=None, now=None):
        policy=policy or NotificationPolicy()
        if not isinstance(policy,NotificationPolicy): raise ValueError('NotificationPolicy required')
        identity=text(event['id'],'event id'); text(destination,'destination')
        payload=json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False)
        if len(payload.encode())>1024*1024: raise ValueError('event exceeds 1 MiB')
        now=time.time() if now is None else number(now,'now',0)
        accepted,reason=policy.eligible(observed_utc=observed_utc,external_facts=external_facts or {},now=now)
        message_id=fingerprint(['behaviorkit-message/1',identity,policy.fingerprint,destination])
        group=fingerprint([policy.fingerprint,destination,event.get('source'),event.get('rule')])
        with closing(self._connect()) as db:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT payload FROM facts WHERE id=?',(identity,)).fetchone()
            if old and old['payload']!=payload: raise ValueError('event ID collision with different payload')
            if not old:
                if db.execute('SELECT count(*) FROM facts').fetchone()[0]>=self.max_facts:
                    raise RuntimeError('fact store capacity exceeded; export/archive facts before retry')
                db.execute('INSERT INTO facts VALUES(?,?,?)',(identity,payload,now))
            existing=db.execute('SELECT reason FROM decisions WHERE id=?',(message_id,)).fetchone()
            if existing:
                db.commit(); return {'message_id':message_id,'reason':existing['reason'],'duplicate':True}
            last=db.execute('SELECT accepted FROM cooldowns WHERE key=?',(group,)).fetchone()
            if accepted and last and now-last['accepted']<policy.cooldown_seconds:
                accepted,reason=False,'cooldown'
            if accepted and db.execute('SELECT count(*) FROM messages WHERE sent IS NULL').fetchone()[0]>=self.max_pending:
                # Commit the raw fact, but leave no decision so enqueue can be retried.
                db.commit(); raise RuntimeError('outbox pending capacity exceeded; fact retained, enqueue must be retried')
            db.execute('INSERT INTO decisions VALUES(?,?,?,?,?)',(message_id,identity,policy.fingerprint,destination,reason))
            if accepted:
                db.execute('INSERT INTO messages(id,event_id,destination,payload,available) VALUES(?,?,?,?,?)',
                           (message_id,identity,destination,payload,now))
                db.execute('INSERT INTO cooldowns VALUES(?,?) ON CONFLICT(key) DO UPDATE SET accepted=excluded.accepted',(group,now))
            db.commit()
        return {'message_id':message_id,'reason':reason,'duplicate':False}

    def claim(self, *, destination='jsonl', limit=100, lease_seconds=30, now=None):
        if type(limit) is not int or not 1<=limit<=1000: raise ValueError('claim limit must be in 1..1000')
        number(lease_seconds,'lease_seconds',.01,3600)
        now=time.time() if now is None else number(now,'now',0)
        token=uuid.uuid4().hex
        with closing(self._connect()) as db:
            db.execute('BEGIN IMMEDIATE')
            rows=db.execute('SELECT * FROM messages WHERE destination=? AND sent IS NULL AND available<=? AND (lease_until IS NULL OR lease_until<=?) ORDER BY available,id LIMIT ?',
                            (destination,now,now,limit)).fetchall()
            for row in rows:
                db.execute('UPDATE messages SET token=?,lease_until=?,attempts=attempts+1 WHERE id=?',(token,now+lease_seconds,row['id']))
            db.commit()
        return tuple({'id':r['id'],'event_id':r['event_id'],'payload':json.loads(r['payload']),
                      'token':token,'attempt':r['attempts']+1} for r in rows)

    def settle(self, message, *, error=None, now=None):
        now=time.time() if now is None else number(now,'now',0)
        with closing(self._connect()) as db:
            if error is None:
                cursor=db.execute('UPDATE messages SET sent=?,token=NULL,lease_until=NULL WHERE id=? AND token=? AND sent IS NULL AND lease_until>?',
                                  (now,message['id'],message['token'],now))
            else:
                delay=min(3600,2**min(12,message['attempt']))
                cursor=db.execute('UPDATE messages SET available=?,token=NULL,lease_until=NULL,last_error=? WHERE id=? AND token=? AND sent IS NULL AND lease_until>?',
                                  (now+delay,str(error)[:1000],message['id'],message['token'],now))
            if cursor.rowcount!=1: raise ValueError('message lease expired or belongs to another dispatcher')

    def dispatch(self, sink, *, destination='jsonl', limit=100, now=None):
        sent=failed=0
        for message in self.claim(destination=destination,limit=limit,now=now):
            try:
                sink(message)
            except Exception as error:
                self.settle(message,error=f'{type(error).__name__}: {error}',now=now); failed+=1
            else:
                self.settle(message,now=now); sent+=1
        return {'sent':sent,'failed':failed,'semantics':'at_least_once; consumers deduplicate message id'}

    def facts(self):
        with closing(self._connect()) as db:
            for row in db.execute('SELECT payload FROM facts ORDER BY created,id'):
                yield json.loads(row[0])

    def status(self):
        with closing(self._connect()) as db:
            return {'facts':db.execute('SELECT count(*) FROM facts').fetchone()[0],
                    'pending':db.execute('SELECT count(*) FROM messages WHERE sent IS NULL').fetchone()[0],
                    'sent':db.execute('SELECT count(*) FROM messages WHERE sent IS NOT NULL').fetchone()[0]}


class JsonlSink:
    def __init__(self,path): self.path=Path(path)
    def __call__(self,message):
        import os
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps({'message_id':message['id'],'event':message['payload']},allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
