from my import *
import os
import csv
from collections import Counter
import multiprocessing as mp
from my.utils import group_by_n, stem, lemma, lemm_or_stem
from my import DataHelper
import math

class WikiPagePreparator:
    # xml -> CSV
    def prepare_wiki_pages(self, reader: DataReader):
        output = os.path.join('data', 'wiki_pages.csv')
        output_wc = os.path.join('data', 'wiki_pages_wc.csv')
        word_counter = Counter()
        with open(output, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['id', 'parentid', 'title', 'words'])
            i = 0
            for page in reader.read_wikipedia_pages():
                words = [w.lower() for w in page.get_cyrillic_words()]
                for w in words:
                    word_counter[w] += 1
                writer.writerow([str(page.id), str(page.parentid), page.title, ",".join(words)])
                i += 1
                if i % 1000 == 0: print("%d pages processed" % i)
            print("pages written")
        with open(output_wc, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['count', 'word'])
            for word, cnt in sorted(word_counter.items(), key=lambda x: x[1], reverse=True)[:10000]:
                writer.writerow([str(cnt), word])
            print("word counter written")

    # CSV (words) -> CSV (lemms - stop_words)
    def prepare_wiki_pages_stems(self, workers=8, stem_name: str = 'stem'):
        csv.field_size_limit(2 ** 31)
        input_filepath = os.path.join('data', 'wiki_pages.csv')
        output_filepath_template = os.path.join('data', 'wiki_pages_%s.%%d.csv' % stem_name)
        output_filepath_final = os.path.join('data', 'wiki_pages_%s.csv' % stem_name)

        prev_position = 0
        chunks = []
        rows_per_chunk = 10000
        with open(os.path.join('data', 'wiki_pages.csv')) as f:
            row_i = 0
            i = 0
            line = 'some'
            while line:
                line = f.readline()
                row_i += 1
                i += 1
                if row_i % 10000 == 0:
                    print('Read file: %d rows' % row_i)
                if i >= rows_per_chunk:
                    position = f.tell()
                    chunks.append((prev_position, position, i))
                    prev_position = position
                    i = 0

        jobs = []
        pool = mp.Pool(workers)
        job_files = []
        for chunks_group in group_by_n(chunks, int(math.ceil(len(chunks) / workers))):
            start = chunks_group[0][0]
            end = chunks_group[-1][-1]
            lines = sum([lines for _, _, lines in chunks_group])
            job_filepath = output_filepath_template % len(jobs)
            job_files.append(job_filepath)
            jobs.append(pool.apply_async(process_wiki_pages_stems_chunk,
                                         (input_filepath, job_filepath, start, end, lines, stem_name)) )

        for job in jobs:
            job.get()
        pool.close()

        with open(output_filepath_final, 'w') as final_file:
            for job_filepath in job_files:
                with open(job_filepath) as f:
                    i = 0
                    for line in f:
                        final_file.write(line)
                        i += 1
                        if i % 10000 == 0:
                            print('Merge file %s: %d rows' % (job_filepath, i))
                os.unlink(job_filepath)

    def prepare_topic_checks(self, reader: DataReader, helper: DataHelper):
        check_corpus = SCTM.corpus(reader.read_sctm())
        topics = [topic for topic, ts in check_corpus.items()]
        raw_docs = [helper.get_lemms([w for t in ts for w in t.words]) for topic, ts in check_corpus.items()]
        topic_words = dict()
        for i, blob in enumerate(raw_docs):
            topic = topics[i]
            print("Top words in document {}".format(i + 1))
            scores = {word: DataHelper.tfidf(word, blob, raw_docs) for word in blob}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            topic_words[topic] = list()
            for word, score in sorted_words:
                print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
                topic_words[topic].append((word, score))
        if not os.path.isdir(os.path.join('data', 'topics')):
            os.mkdir(os.path.join('data', 'topics'))
        for topic, words in topic_words.items():
            with open(os.path.join('data', 'topics', topic + '.txt'), 'w') as f:
                for w, _ in words:
                    f.write(w + '\n')


def process_wiki_pages_stems_chunk(input_filepath, output_filepath, start, end, lines, stem_name):
    print("I should read %d-%d and write to %s" % (start, end, output_filepath))
    stem_cb = None
    if 'lemm' in stem_name and 'stem' in stem_name:
        stem_cb = lemm_or_stem
    if 'stem' in stem_name:
        stem_cb = stem
    elif 'lemm' in stem_name:
        stem_cb = lemma
    else:
        raise RuntimeError("Undefined stemmer type: %s" % stem_name)
    min_stems = 10
    max_stems = 1000

    stop_words = DataReader().read_stop_words()
    with open(output_filepath, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        with open(input_filepath) as f:
            f.seek(start)
            i = 0
            for line in f:
                pageid, parentid, title, words = line.split('\t')
                words = [w for w in words.strip().split(',') if not w in stop_words and len(w) > 2]
                if len(words) >= min_stems:
                    stems = [stem_cb(w) for w in words]
                    writer.writerow([pageid, parentid, title, ','.join(stems[:max_stems])])
                i += 1
                if i % 10000 == 0:
                    print('Handle file %s: %d rows' % (output_filepath, i))
                if i >= lines:
                    break