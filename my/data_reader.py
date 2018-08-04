from typing import List, Iterator, Set
from my.model import *
from my.utils import stem, lemma
import os
import json
import bz2
from lxml import etree
import csv

class DataReader:
    def read_classic_poems(self) -> List[Poem]:
        poems: List[Poem] = []
        with open(os.path.join('data', 'classic_poems.json')) as f:
            for entry in json.load(f):
                poem: Poem = Poem(Poet.by_poet_id(entry['poet_id']), entry['title'], entry['content'])
                poems.append(poem)
        return poems

    def read_opcorpora(self) -> Iterator[str]:
        pass

    def read_wikipedia_pages(self) -> Iterator[WikiPage]:
        with bz2.open(os.path.join('data', 'ruwiki-latest-pages-articles-multistream.xml.bz2')) as f:
            #path = []
            tag_id = None
            tag_parent_id = None
            tag_title = None
            tag_text = None
            skip_page = False
            namespace = '{http://www.mediawiki.org/xml/export-0.10/}'
            for event, element in etree.iterparse(f,
                                                  events=('start', 'end'),
                                                  tag=(namespace + 'page', namespace + 'id', namespace + 'parentid',
                                                       namespace + 'title', namespace + 'text', namespace + 'redirect'),
                                                  encoding='utf-8'):
                # print(' %s/%s' % (repr(event), repr(tag)))
                tag = element.tag.replace(namespace, '')
                if event == 'start':
                    #path.append(tag)
                    if tag == 'page':
                        tag_id = None
                        tag_parent_id = None
                        tag_title = None
                        tag_text = None
                        skip_page = False
                elif event == 'end':
                    if tag == 'page':
                        if tag_text and not skip_page:
                            yield WikiPage(tag_id, tag_parent_id, tag_title, content=tag_text)
                    elif not skip_page:
                        if tag == 'id':
                            tag_id = int(element.text)
                        elif tag == 'parentid':
                            tag_parent_id = int(element.text)
                        elif tag == 'title':
                            tag_title = element.text
                            skip_page = ':' in tag_title
                        elif tag == 'text':
                            tag_text = element.text
                        elif tag == 'redirect':
                            skip_page = True
                    element.clear()

    def read_wikipedia_pages2(self,
                              stemming: bool = False,
                              lemmaatazing: bool = False) -> Iterator[WikiPage2]:
        csv.field_size_limit(2 ** 31)
        with open(os.path.join('data', 'wiki_pages.csv')) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            next(reader) # skip header
            for row in reader:
                id = int(row[0])
                parentid = None
                if row[1] != 'None':
                    parentid = int(row[1])
                title = row[2]
                words = row[3].split(',')
                if stemming:
                    words = [stem(w) for w in words]
                if lemmaatazing:
                    words = [lemma(w) for w in words]
                yield WikiPage2(id, parentid, title, words)

    def read_stop_words(self) -> Set[str]:
        result = None
        with open(os.path.join('data', 'stop_words.csv')) as f:
            result = [l for l in f.readlines() if l]
        return set(result)
