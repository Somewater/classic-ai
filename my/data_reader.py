from typing import List, Iterator, Set, Dict
from my.model import *
from my.utils import stem, lemma
import os
import json
import bz2
from lxml import etree
import csv
import re

BigLetters = re.compile('[A-Z]+')

class DataReader:
    def read_classic_poems(self) -> List[Poem]:
        poems: List[Poem] = []
        with open(os.path.join('data', 'classic_poems.json')) as f:
            for entry in json.load(f):
                poem: Poem = Poem(Poet.by_poet_id(entry['poet_id']), entry['title'], entry['content'])
                poems.append(poem)
        return poems

    def read_opcorpora(self, ignore_gs: bool = True) -> Iterator[OpCorpText]:
        with bz2.open(os.path.join('data', 'annot.opcorpora.xml.bz2')) as f:
            tag_text_id = None
            tag_text_title = None
            tag_text_paragraphs = None
            tag_paragraph_sentenses = None
            tag_sentense_id = None
            tag_sentense_source = None
            tag_sentense_tokens = None
            tag_token_id = None
            tag_token_text = None
            tag_token_gs = None

            for event, element in etree.iterparse(f, events=('start', 'end'), encoding='utf-8'):
                tag = element.tag
                if event == 'start':
                    if tag == 'text':
                        tag_text_id = int(element.attrib['id'])
                        tag_text_title = element.attrib['name']
                        tag_text_paragraphs = None
                        tag_paragraph_sentenses = None
                        tag_sentense_id = None
                        tag_sentense_source = None
                        tag_sentense_tokens = None
                        tag_token_id = None
                        tag_token_text = None
                        tag_token_gs = None
                    elif tag == 'paragraphs':
                        tag_text_paragraphs = []
                    elif tag == 'paragraph':
                        tag_paragraph_sentenses = []
                    elif tag == 'sentence':
                        tag_sentense_id = int(element.attrib['id'])
                    elif tag == 'tokens':
                        tag_sentense_tokens = []
                    elif tag == 'token':
                        tag_token_id = int(element.attrib['id'])
                        tag_token_text = element.attrib['text']
                    elif tag == 'tfr':
                        tag_token_gs = []
                elif event == 'end':
                    if tag == 'source':
                        tag_sentense_source = element.text
                    elif tag == 'g':
                        tag_token_gs.append(element.attrib['v'])
                    elif tag == 'token':
                        pos = [g for g in tag_token_gs if BigLetters.fullmatch(g)]
                        if pos:
                            pos = pos[0]
                        else:
                            pos = None
                        if ignore_gs:
                            tag_token_gs = None
                        tag_sentense_tokens.append(OpCorpToken(tag_token_id, tag_token_text, tag_token_gs, pos))
                        tag_token_gs = None
                    elif tag == 'sentence':
                        tag_paragraph_sentenses.append(OpCorpSentence(tag_sentense_id, tag_sentense_source, tag_sentense_tokens))
                        tag_sentense_tokens = None
                    elif tag == 'paragraph':
                        tag_text_paragraphs.append(OpCorpParagraph(tag_paragraph_sentenses))
                        tag_paragraph_sentenses = None
                    elif tag == 'text':
                        if tag_text_paragraphs:
                            yield OpCorpText(tag_text_id, tag_text_title, tag_text_paragraphs)
                        tag_text_paragraphs = None
                    element.clear()

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
                              lemmaatazing: bool = False,
                              filename: str = 'wiki_pages.csv') -> Iterator[WikiPage2]:
        csv.field_size_limit(2 ** 31)
        with open(os.path.join('data', filename)) as csvfile:
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

    def get_tmp_filepath(self, filename: str = None):
        if filename:
            return os.path.join('tmp', filename)
        else:
            return os.path.join('tmp')

    def load_word_count(self, filepath: str = 'data/word_count.csv') -> Dict[str, int]:
        wc = dict()
        with open(filepath) as file:
            for line in file:
                cnt = int(line[:10])
                text = line[11:].strip()
                wc[text] = cnt
        return wc

    def save_word_count(self, wc: Dict[str, int], filepath: str = 'data/word_count.csv'):
        with open(filepath, 'w') as file:
            for w, c in sorted(wc.items(), key=lambda x: (-x[1], x[0].lower())):
                line = "%10.d %s\n" % (c, w)
                file.write(line)

class OpCorpus(Corpus):
    def __init__(self, reader: DataReader):
        super().__init__('opcorpora', reader.read_opcorpora)

class WikiCorpus(Corpus):
    def __init__(self, reader: DataReader, type: str = ''):
        filename = 'wiki_pages.csv'
        if type:
            filename = 'wiki_pages_%s.csv' % type
        super().__init__('wiki_corpus', reader.read_wikipedia_pages2, False, False, filename)