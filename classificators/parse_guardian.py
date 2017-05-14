from bs4 import BeautifulSoup, SoupStrainer

from classificators.common import *

import urllib.request
import codecs


def get_article_links(url, number_of_pages=100):
    links = set()

    while number_of_pages > 0:
        print(number_of_pages)
        link = urllib.request.urlopen(url + str(number_of_pages))
        response = link.read()
        link.close()
        response = response.decode("utf-8")
        soup = BeautifulSoup(response, "html.parser",
                             parse_only=SoupStrainer("a", attrs={'data-link-name': 'article'}))
        for link in soup:
            if link.has_attr('href'):
                links.add(link['href'])
        number_of_pages -= 1

    return links


def get_article_from_link(link_str):
    link = urllib.request.urlopen(link_str)
    response = link.read()
    link.close()
    response = response.decode("utf-8")
    soup = BeautifulSoup(response, "html.parser",
                         parse_only=SoupStrainer("div", attrs={'itemprop': 'articleBody'}))
    return soup.text.replace('\n', '').replace('Read more', '')


def save_article_to_file(article, filename, directory):
    file = codecs.open(join(directory, filename), 'w', "utf-8")
    file.write(article)
    file.close()


def parse_and_save(url, number_of_pages, directory_to_write):
    links = get_article_links(url, number_of_pages)
    i = 0

    files = get_files_from(directory_to_write)

    while (str(i) + ".txt") in files:
        i += 1

    for link in links:
        print(link)

        i += 1

        while (str(i) + ".txt") in files:
            i += 1

        try:
            article = get_article_from_link(link)
            save_article_to_file(article, str(i) + ".txt", directory_to_write)
        except Exception as e:
            print(e.strerror)

#parse_and_save("https://www.theguardian.com/uk/ukcrime?page=", 734, join("..", "the_guardian_com", "crimes"))#
#parse_and_save("https://www.theguardian.com/lifeandstyle?page=", 300, join("..", "the_guardian_com", "not_crimes"))
#parse_and_save("https://www.theguardian.com/technology?page=", 300, join("..", "the_guardian_com", "not_crimes"))
#parse_and_save("https://www.theguardian.com/uk/sport?page=", 400, join("..", "the_guardian_com", "not_crimes"))
#parse_and_save("https://www.theguardian.com/uk/culture?page=", 1000, join("..", "the_guardian_com", "not_crimes"))
parse_and_save("https://www.theguardian.com/uk/environment?page=", 1000, join("..", "the_guardian_com", "not_crimes"))
