import tempfile
from pathlib import Path
import pytest
from pipeline.news.parsers import parse_file


PROQUEST_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<RECORDS>
  <Record>
    <Title>Apple Reports Record Earnings</Title>
    <FullText>Apple Inc reported record quarterly earnings on Tuesday, beating analyst expectations.</FullText>
    <PublicationTitle>Wall Street Journal</PublicationTitle>
    <PubDate>2024-01-16</PubDate>
    <Authors>Jane Smith; John Doe</Authors>
    <DocumentID>PQ-20240116-001</DocumentID>
    <StartPage>A1</StartPage>
    <PublicationType>Markets</PublicationType>
  </Record>
  <Record>
    <Title>Tesla Deliveries Miss Estimates</Title>
    <FullText>Tesla delivered fewer vehicles than expected in the fourth quarter.</FullText>
    <PublicationTitle>Wall Street Journal</PublicationTitle>
    <PubDate>2024-01-15</PubDate>
    <Authors>Bob Jones</Authors>
    <DocumentID>PQ-20240115-001</DocumentID>
  </Record>
</RECORDS>
"""

FACTIVA_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<Results>
  <doc>
    <an>NYT-20240115-001</an>
    <hd>Fed Signals Rate Pause</hd>
    <sn>New York Times</sn>
    <pd>January 15, 2024</pd>
    <by>Alice Reporter</by>
    <td><p>The Federal Reserve signaled it may pause interest rate hikes.</p></td>
  </doc>
</Results>
"""

CSV_DATA = """article_id,source,headline,full_body,publication_date,publication_time,section,front_page,authors,database_source_id
art001,WSJ,Nike Posts Strong Quarter,Nike beat quarterly revenue estimates driven by direct-to-consumer sales.,2024-02-01,09:30:00,Markets,true,Reporter One,DB001
art002,NYT,Walmart Raises Guidance,Walmart raised its annual guidance citing strong consumer demand.,2024-02-02,,,Business,false,Reporter Two,DB002
"""


def write_tmp(content, suffix, tmp_path):
    p = tmp_path / f"test{suffix}"
    p.write_text(content, encoding="utf-8")
    return p


def test_proquest_xml_parse(tmp_path):
    p = write_tmp(PROQUEST_XML, ".xml", tmp_path)
    articles = parse_file(p, source_hint="WSJ")
    assert len(articles) == 2
    assert articles[0].source == "WSJ"
    assert "Apple" in articles[0].headline
    assert articles[0].front_page is True  # A1 page
    assert "Jane Smith" in articles[0].authors
    assert articles[0].word_count > 0


def test_factiva_xml_parse(tmp_path):
    p = write_tmp(FACTIVA_XML, ".xml", tmp_path)
    articles = parse_file(p, source_hint="NYT", fmt="factiva")
    assert len(articles) == 1
    assert articles[0].source == "NYT"
    assert "Fed" in articles[0].headline
    assert articles[0].database_source_id == "NYT-20240115-001"


def test_csv_parse(tmp_path):
    p = write_tmp(CSV_DATA, ".csv", tmp_path)
    articles = parse_file(p, source_hint="WSJ")
    assert len(articles) == 2
    assert articles[0].source == "WSJ"
    assert articles[0].front_page is True
    assert articles[1].front_page is False
    assert articles[0].publication_date == "2024-02-01"


def test_csv_word_count(tmp_path):
    p = write_tmp(CSV_DATA, ".csv", tmp_path)
    articles = parse_file(p)
    assert all(a.word_count > 0 for a in articles)


def test_empty_records_skipped(tmp_path):
    xml = '<?xml version="1.0"?><RECORDS><Record><Title></Title><FullText></FullText></Record></RECORDS>'
    p = write_tmp(xml, ".xml", tmp_path)
    articles = parse_file(p)
    assert len(articles) == 0
