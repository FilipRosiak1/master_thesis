import nltk
import numpy as np
import six


gram = """S -> SEQ
SEQ -> ELEMENT SEQ
SEQ -> ELEMENT
ELEMENT -> MODS CORE
CORE -> 'X'
CORE -> '(' BRANCHES ')'
BRANCHES -> BRANCH BRANCH_TAIL
BRANCHES ->
BRANCH_TAIL -> ',' BRANCH BRANCH_TAIL
BRANCH_TAIL -> ',' BRANCH_TAIL
BRANCH_TAIL ->
BRANCH -> SEQ
BRANCH ->
MODS -> MODIFIER MODS
MODS ->
MODIFIER -> 'R'
MODIFIER -> 'r'
MODIFIER -> 'Q'
MODIFIER -> 'q'
MODIFIER -> 'C'
MODIFIER -> 'c'
MODIFIER -> 'L'
MODIFIER -> 'l'
MODIFIER -> 'W'
MODIFIER -> 'w'
MODIFIER -> 'M'
MODIFIER -> 'm'
MODIFIER -> 'I'
MODIFIER -> 'i'
MODIFIER -> 'F'
MODIFIER -> 'f'
MODIFIER -> 'A'
MODIFIER -> 'a'
MODIFIER -> 'S'
MODIFIER -> 's'
MODIFIER -> 'E'
MODIFIER -> 'e'
Nothing -> None"""

GCFG = nltk.CFG.fromstring(gram)
start_index = GCFG.productions()[0].lhs()

all_lhs = [prod.lhs().symbol() for prod in GCFG.productions()]
lhs_list = []
for lhs in all_lhs:
    if lhs not in lhs_list:
        lhs_list.append(lhs)

D = len(GCFG.productions())

rhs_map = [None] * D
for prod_ix, prod in enumerate(GCFG.productions()):
    rhs_map[prod_ix] = []
    for sym in prod.rhs():
        if not isinstance(sym, six.string_types):
            symbol = sym.symbol()
            rhs_map[prod_ix].extend(list(np.where(np.array(lhs_list) == symbol)[0]))

masks = np.zeros((len(lhs_list), D))
for lhs_ix, lhs_symbol in enumerate(lhs_list):
    masks[lhs_ix] = np.array([entry == lhs_symbol for entry in all_lhs], dtype=int).reshape(1, -1)

index_array = []
for i in range(masks.shape[1]):
    index_array.append(np.where(masks[:, i] == 1)[0][0])

ind_of_ind = np.array(index_array)
