from datetime import datetime
from xml.sax.saxutils import escape as xml_escape


def timestamped_filename(stem: str, ext: str = "png") -> str:
    return f"{datetime.now():%Y-%m-%d-%H%M%S}-{stem}.{ext}"

def last_path_component(p: str) -> str:
    return p.split("/")[-1].split(".")[0]

def xmp_description_packet(description: str) -> str:
    safe = xml_escape(description, {'"': "&quot;", "'": "&apos;"})
    return (
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">'
        '<dc:description><rdf:Alt>'
        f'<rdf:li xml:lang="x-default">{safe}</rdf:li>'
        '</rdf:Alt></dc:description>'
        '</rdf:Description>'
        '</rdf:RDF>'
        '</x:xmpmeta>'
        '<?xpacket end="w"?>'
    )
