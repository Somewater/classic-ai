from typing import List, Iterator
from my.model import *
import os
import json
import bz2
from lxml import etree

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
            system_page = False
            namespace = '{http://www.mediawiki.org/xml/export-0.10/}'
            for event, element in etree.iterparse(f,
                                                  events=('start', 'end'),
                                                  tag=(namespace + 'page', namespace + 'id', namespace + 'parentid',
                                                       namespace + 'title', namespace + 'text'),
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
                        system_page = False
                elif event == 'end':
                    if tag == 'page':
                        if tag_text:
                            yield WikiPage(tag_id, tag_parent_id, tag_title, content=tag_text)
                    elif not system_page:
                        if tag == 'id':
                            tag_id = int(element.text)
                        elif tag == 'parentid':
                            tag_parent_id = int(element.text)
                        elif tag == 'title':
                            tag_title = element.text
                            system_page = ':' in tag_title
                        elif tag == 'text':
                            tag_text = element.text
                    element.clear()
