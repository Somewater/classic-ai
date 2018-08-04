from my import DataReader
import os
import csv
from collections import Counter

class WikiPagePreparator:
    def prepare(self, reader: DataReader):
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
