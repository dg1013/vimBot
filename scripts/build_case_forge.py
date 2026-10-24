#!/usr/bin/env python3
"""Build the Case Forge artifact: hand-author new v3 cases against the gaps.

The browser cannot run vim, so the page and scripts/ingest_handcrafted.py
split the work honestly: the page drafts cases and enforces everything
checkable statically (§6 instruction rules, the leak guard, cursor bounds,
in-batch duplicates), stores drafts in the browser, and exports a JSON (JavaScript Object Notation) list;
the ingest script executes every draft in the sandboxed executor and is the
only path into data/v3/handcrafted_cases.jsonl.

Seeds: every starter example on the page is executed through run_vim AT BUILD
TIME and embedded with vim's actual output — the build fails if a seed stops
working, so the gallery can never show a command that does not run.

Targets: quota fill per category and per VimGolf theme is computed from the
live corpus, so the "aim here" chips stay current. Rebuild after ingests:

    conda run -n vimBot python scripts/build_case_forge.py --out page.html
"""
import argparse
import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from executor import run_vim  # noqa: E402
sys.path.insert(0, str(REPO / "scripts"))
from page_guard import check_page  # noqa: E402

try:
    import yaml
except ImportError:
    sys.exit("needs pyyaml (present in the vimBot env)")

# label, category, tier, theme, mode, command (ex: no leading colon),
# buffer_before, cursor, instruction suggestion (§6 voice; user edits freely)
SEEDS = [
    # (label, category, theme, mode, command, buffer_before, cursor, hint)
    ("Swap Last, First", "substitute", "T01", "ex",
     r"%s/\(\w\+\), \(\w\+\)/\2 \1/", "Doe, Jane\nRoe, Richard", [1, 1],
     "Turn each Last, First name into First Last order"),
    ("Quote every line", "substitute", "", "ex",
     r"%s/^\(.*\)$/\"\1\"/", "alpha\nbeta\ngamma", [1, 1],
     "Wrap every line in double quotes"),
    ("Title-case words", "substitute", "T01", "ex",
     r"%s/\<\(\w\)\(\w*\)\>/\u\1\2/g", "the quick brown fox", [1, 1],
     "Capitalise the first letter of every word"),
    ("Double the numbers", "substitute", "T07", "ex",
     r"%s/\d\+/\=submatch(0)*2/g", "buy 3 apples and 12 pears", [1, 1],
     "Double every number on the line"),
    ("CSV to markdown", "substitute", "T02", "ex",
     r"%s/^\(.*\),\(.*\),\(.*\)$/| \1 | \2 | \3 |/",
     "jan,31,cold\nfeb,28,wet", [1, 1],
     "Turn each comma-separated row into a markdown table row"),
    ("YAML to dotenv", "substitute", "T02", "ex",
     r"%s/^\(\w\+\): \(.*\)$/\U\1\E=\2/",
     "host: example.com\nport: 8080", [1, 1],
     "Rewrite each key: value line as an uppercase KEY=value line"),
    ("dotenv to JSON", "substitute", "T02", "ex",
     r"%s/^\(\w\+\)=\(.*\)$/  \"\1\": \"\2\",/",
     "user=jane\nrole=admin", [1, 1],
     "Rewrite each KEY=value line as an indented JSON member"),
    ("Python def to JS", "substitute", "T03", "ex",
     r"%s/^def \(\w\+\)(\(.*\)):$/function \1(\2) {/",
     "def add(a, b):", [1, 1],
     "Convert the Python function header to JavaScript syntax"),
    ("Align the equals", "substitute", "T05", "ex",
     r"%s/^\(\w\+\) *=/\=printf('%-8s=', submatch(1))/",
     "x = 1\nlongname = 2\nab = 3", [1, 1],
     "Pad the variable names so the equals signs line up"),
    ("Split at commas", "substitute", "T10", "ex",
     r"%s/,/,\r/g", "red,green,blue", [1, 1],
     "Break the list so each item ends its own line"),
    ("Squash rebase picks", "substitute", "T12", "ex",
     "2,3s/^pick/squash/",
     "pick 1a2b3c first\npick 4d5e6f second\npick 7a8b9c third", [1, 1],
     "Mark the second and third commits to be squashed"),
    ("Bump the ssh port", "substitute", "T12", "ex",
     r"%s/^Port \d\+$/Port 2222/", "Host dev\nPort 22\nUser jane", [1, 1],
     "Change the port setting to 2222"),
    ("Minor version bump", "substitute", "T12", "ex",
     r"s/\(version = \"0\.\)\(\d\+\)\.\d\+\"/\=submatch(1).(submatch(2)+1).'.0\"'/",
     'version = "0.4.2"', [1, 1],
     "Raise the minor version by one and reset the patch number"),
    ("Fix stuck keys", "substitute", "T15", "ex",
     r"%s/\(\w\)\1\+/\1/g", "It'ss tooo coold too typpe", [1, 1],
     "Collapse every run of repeated letters to a single one"),
    ("Sentence case", "substitute", "T15", "ex",
     r"%s/\. \(\w\)/. \u\1/g", "one. two. three.", [1, 1],
     "Capitalise the first letter after each full stop"),
    ("Lookahead match", "substitute", "T17", "ex",
     r"%s/foo\(bar\)\@=/X/g", "foobar foobaz foobar", [1, 1],
     "Replace foo with X, but only where bar follows it"),
    ("Normalise a word's case", "substitute", "T17", "ex",
     r"%s/\c\<vim\>/Vim/g", "vim and VIM and vIm", [1, 1],
     "Spell every occurrence of the editor's name as Vim"),

    ("Delete the TODO lines", "global", "T14", "ex",
     "g/TODO/d", "keep this\nTODO drop\nalso keep\nTODO gone", [1, 1],
     "Delete every line containing TODO"),
    ("Keep only matches", "global", "T14", "ex",
     "v/cat/d", "cat one\ndog two\ncat three\nbird", [1, 1],
     "Keep only the lines that mention cat"),
    ("Reverse the file", "global", "T09", "ex",
     "g/^/m0", "first\nsecond\nthird", [1, 1],
     "Reverse the order of the lines"),
    ("Tag the error lines", "global", "T14", "ex",
     "g/error/s/$/ <-- check/", "ok\nerror one\nfine\nerror two", [1, 1],
     "Append a check marker to every line containing error"),
    ("Strip conflict markers", "global", "T20", "ex",
     r"g/^[<=>]\{7}/d",
     "keep\n<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> branch\ntail", [1, 1],
     "Remove the merge conflict markers, keeping both versions"),

    ("Move a line to the top", "move-copy", "T09", "ex",
     "2m0", "alpha\nbeta\ngamma", [1, 1],
     "Move the second line above the first"),
    ("Copy two lines to the end", "move-copy", "", "ex",
     "1,2t$", "one\ntwo\nthree", [1, 1],
     "Append a copy of the first two lines to the end"),
    ("Shuffle a line down", "move-copy", "T09", "ex",
     "3m1", "aa\nbb\ncc\ndd", [1, 1],
     "Move the third line so it sits below the first"),

    ("Sort alphabetically", "sort-uniq", "T06", "ex",
     "sort", "pear\napple\nmango", [1, 1],
     "Sort the lines alphabetically"),
    ("Sort and dedup", "sort-uniq", "T06", "ex",
     "sort u", "bb\naa\nbb\ncc\naa", [1, 1],
     "Sort the lines and drop the duplicates"),
    ("Numeric sort of a range", "sort-uniq", "T06", "ex",
     "2,4sort n", "head\n30\n4\n200\ntail", [1, 1],
     "Sort the three numbers numerically, leaving the first and last lines"),

    ("Join three lines", "join", "T10", "keys",
     "3J", "one\ntwo\nthree\nfour", [1, 1],
     "Join the first three lines with spaces"),
    ("Join without a space", "join", "T10", "keys",
     "gJ", "left\nright", [1, 1],
     "Join the two lines with nothing between them"),
    ("Flatten the file", "join", "T10", "ex",
     "%j", "aa\nbb\ncc", [1, 1],
     "Join every line into a single line"),
    ("Join a range", "join", "T10", "ex",
     "2,3j", "keep\nfoo\nbar\nend", [1, 1],
     "Join the second and third lines together"),

    ("Delete the middle lines", "line-ops", "", "ex",
     "2,3d", "top\nmid1\nmid2\nbottom", [1, 1],
     "Delete the two middle lines"),
    ("Swap two lines", "line-ops", "T09", "keys",
     "ddp", "swap1\nswap2\nrest", [1, 1],
     "Swap the first two lines"),
    ("Duplicate upward", "line-ops", "T16", "keys",
     "yyP", "solo\nnext", [1, 1],
     "Insert a copy of the first line above it"),

    ("Delete until a comma", "operator-motion", "T04", "keys",
     "dt,", "delete this,keep this", [1, 1],
     "Delete everything before the first comma"),
    ("Cut to end of line", "operator-motion", "", "keys",
     "d$", "keep tail cut here", [1, 6],
     "Delete from the cursor to the end of the line"),
    ("Replace the first word", "operator-motion", "T04", "keys",
     "cwnew<Esc>", "old word stays", [1, 1],
     "Replace the first word with new"),

    ("Empty the parentheses", "text-objects", "T04", "keys",
     "di(", "call(remove me)", [1, 6],
     "Clear everything inside the parentheses"),
    ("Delete a quoted part", "text-objects", "T11", "keys",
     'da"', 'say "quoted" end', [1, 5],
     "Delete the quoted text along with its quotes"),
    ("Retitle a tag", "text-objects", "T04", "keys",
     "citnew<Esc>", "<b>old</b>", [1, 4],
     "Replace the tag's contents with new"),
    ("Drop a paragraph", "text-objects", "", "keys",
     "dap", "para one\n\npara two\ntail", [1, 1],
     "Delete the first paragraph and the blank line after it"),

    ("Shout the first word", "case-ops", "T01", "keys",
     "gUiw", "shout now", [1, 1],
     "Uppercase the first word"),
    ("Toggle a whole line", "case-ops", "T01", "keys",
     "g~~", "MiXeD line", [1, 1],
     "Flip the case of every character on the line"),
    ("Hush a whole line", "case-ops", "T01", "keys",
     "guu", "LOUD TEXT", [1, 1],
     "Lowercase the entire line"),

    ("Indent one line", "indent", "T13", "keys",
     ">>", "move me\nstay put", [1, 1],
     "Indent the first line by one level"),
    ("Indent three lines", "indent", "T13", "keys",
     "3>>", "aa\nbb\ncc\ndd", [1, 1],
     "Indent the first three lines"),
    ("Outdent a range", "indent", "T13", "ex",
     "2,3<", "\tkeep\n\tshift\n\tshift too", [1, 1],
     "Remove one level of indentation from the second and third lines"),

    ("Add five", "increment", "T07", "keys",
     "5<C-a>", "count 10", [1, 1],
     "Add five to the number"),
    ("Step down once", "increment", "T07", "keys",
     "<C-x>", "level 3", [1, 1],
     "Decrease the number by one"),
    ("Grow a numbered list", "increment", "T08", "keys",
     "yyp<C-a>yyp<C-a>", "step 1", [1, 1],
     "Extend the list with steps two and three"),

    ("Three chars, one at a time", "repeat-count", "", "keys",
     "x..", "abcde", [1, 1],
     "Delete the first three characters"),
    ("Two words via repeat", "repeat-count", "", "keys",
     "dw.", "one two three rest", [1, 1],
     "Delete the first two words"),
    ("Two lines at once", "repeat-count", "", "keys",
     "2dd", "l1\nl2\nl3", [1, 1],
     "Delete the first two lines"),

    ("Delete up to a word", "search-edit", "", "keys",
     "d/wolf\n", "sheep sheep wolf sheep", [1, 1],
     "Delete everything before the wolf"),
    ("Rewrite from a match", "search-edit", "", "keys",
     "/error\nC fixed", "ok line\nan error here", [1, 1],
     "Find the word error and replace it and the rest of its line with fixed"),
    ("Cut the tail after a match", "search-edit", "", "keys",
     "/beta\nD", "alpha\nkeep beta gone", [1, 1],
     "Delete from the word beta to the end of its line"),

    ("Strip trailing blanks", "whitespace", "T13", "ex",
     r"%s/\s\+$//", "clean\ntrailing   \n  both\t\t", [1, 1],
     "Remove the whitespace at the end of each line"),
    ("Squeeze blank runs", "whitespace", "T13", "ex",
     r"%s/\n\n\+/\r/", "aa\n\n\n\nbb\n\ncc", [1, 1],
     "Remove all the blank lines"),
    ("Retab the file", "whitespace", "T13", "ex",
     "retab! 2", "    indented\n        deeper", [1, 1],
     "Convert the space indentation to tabs at a width of two"),

    ("Note below the first line", "insert-edit", "", "keys",
     "ofound it<Esc>", "start here", [1, 1],
     "Add a line reading found it below the first"),
    ("Prefix the line", "insert-edit", "", "keys",
     "Istart <Esc>", "middle end", [1, 1],
     "Put the word start at the beginning of the line"),
    ("Punctuate the end", "insert-edit", "", "keys",
     "A!<Esc>", "wow", [1, 1],
     "Add an exclamation mark at the end of the line"),
    ("Open a line above", "insert-edit", "", "keys",
     "Otop<Esc>", "body text", [1, 1],
     "Add a line reading top above the first"),
    ("Bullet the line", "insert-edit", "", "keys",
     "i- <Esc>", "task", [1, 1],
     "Turn the line into a dash bullet"),

    ("Flip one letter", "case-ops", "T01", "keys",
     "~", "flip me", [1, 1],
     "Flip the case of the first character"),
    ("Quiet the tail", "case-ops", "T01", "keys",
     "gu$", "KEEP QUIET NOW", [1, 6],
     "Lowercase everything from the cursor to the end of the line"),
    ("Yell the line", "case-ops", "T01", "ex",
     r"%s/.*/\U&/", "quiet words", [1, 1],
     "Uppercase the entire line"),

    ("Add ten", "increment", "T07", "keys",
     "10<C-a>", "price 90", [1, 1],
     "Add ten to the number"),
    ("Step up three times", "increment", "T07", "keys",
     "<C-a><C-a><C-a>", "v1 draft", [1, 1],
     "Raise the version number by three"),

    ("Indent everything", "indent", "T13", "ex",
     "%>", "one\ntwo", [1, 1],
     "Indent every line by one level"),
    ("Indent this and the next", "indent", "T13", "keys",
     ">j", "aa\nbb\ncc", [1, 1],
     "Indent the first two lines"),

    ("Join two plainly", "join", "T10", "keys",
     "J", "up\ndown", [1, 1],
     "Join the second line onto the first with a space"),

    ("Delete the opener", "line-ops", "", "ex",
     "1d", "first\nsecond", [1, 1],
     "Delete the first line"),
    ("Cut to the end", "line-ops", "", "keys",
     "dG", "keep\ncut this\ncut too", [2, 1],
     "Delete from the second line to the end of the file"),

    ("Send a line to the end", "move-copy", "T09", "ex",
     "m$", "move me\nstay\nalso stay", [1, 1],
     "Move the first line to the end of the file"),
    ("Copy upward", "move-copy", "T16", "ex",
     "2t0", "aa\nbb\ncc", [1, 1],
     "Put a copy of the second line at the very top"),

    ("Delete through a letter", "operator-motion", "T04", "keys",
     "dfx", "abcxdef", [1, 1],
     "Delete the first characters up to and including the x"),
    ("Clear back to the start", "operator-motion", "", "keys",
     "d0", "prefixkeep", [1, 7],
     "Delete everything before the cursor on this line"),

    ("Three at a stroke", "repeat-count", "", "keys",
     "3x", "dropme", [1, 1],
     "Delete the first three letters of the word"),
    ("Paste three copies", "repeat-count", "T16", "keys",
     "yy3p", "law\nend", [1, 1],
     "Make three more copies of the first line below it"),

    ("Change up to a word", "search-edit", "", "keys",
     "c/end\nX<Esc>", "start middle end", [1, 1],
     "Replace everything before the word end with X"),
    ("Find and delete a word", "search-edit", "", "keys",
     "/gone\ndaw", "keep gone keep", [1, 1],
     "Find the word gone and delete it"),

    ("Reverse sort", "sort-uniq", "T06", "ex",
     "sort!", "aa\ncc\nbb", [1, 1],
     "Sort the lines in reverse order"),
    ("Sort ignoring case", "sort-uniq", "T06", "ex",
     "sort i", "Beta\nalpha\nGamma", [1, 1],
     "Sort the lines alphabetically, ignoring case"),

    ("Empty the brackets", "text-objects", "T11", "keys",
     "ci[", "list[old contents]", [1, 6],
     "Clear the contents of the square brackets"),

    ("Drop the blank-ish lines", "whitespace", "T13", "ex",
     r"g/^\s*$/d", "aa\n   \n\nbb", [1, 1],
     "Remove every blank or whitespace-only line"),
    ("Single-space the words", "whitespace", "T13", "ex",
     r"%s/  \+/ /g", "too  many   spaces here", [1, 1],
     "Collapse every run of spaces into a single space"),

    ("CSV to YAML", "substitute", "T02", "ex",
     r"%s/^\(\w\+\),\(.*\)$/\1: \2/", "name,jane\ncity,oslo", [1, 1],
     "Turn each comma pair into a key and value line"),
    ("List to checklist", "substitute", "T02", "ex",
     r"%s/^- /- [ ] /", "- eggs\n- milk\n- bread", [1, 1],
     "Turn each list item into an unchecked task box"),

    ("print to console.log", "substitute", "T03", "ex",
     r"%s/^print(\(.*\))$/console.log(\1);/", "print(\"hello\")", [1, 1],
     "Convert the Python print call to JavaScript"),
    ("Hash to slash comments", "substitute", "T03", "ex",
     r"%s@^# @// @", "# a note\ncode()\n", [1, 1],
     "Convert the hash comment to a double-slash comment"),
    ("snake_case to camelCase", "substitute", "T03", "ex",
     r"%s/_\(\w\)/\u\1/g", "user_first_name = 1", [1, 1],
     "Convert the snake case name to camel case"),
    ("var to let", "substitute", "T03", "ex",
     r"%s/\<var\>/let/g", "var x = 1; var y = 2;", [1, 1],
     "Modernise the declarations from var to let"),

    ("Right-align the numbers", "substitute", "T05", "ex",
     r"%s/\d\+$/\=printf('%5d', submatch(0))/", "a 1\nbb 22\nccc 333", [1, 1],
     "Right-align the trailing numbers in a five-wide column"),
    ("Pad the first column", "substitute", "T05", "ex",
     r"%s/^\(\w\+\) /\=printf('%-6s ', submatch(1))/", "id 1\nlabel 2", [1, 1],
     "Pad the first column to six characters wide"),
    ("Space out the equals", "substitute", "T05", "ex",
     r"%s/=/ = /", "x=1\ny=2", [1, 1],
     "Put spaces around each equals sign"),
    ("One space after commas", "substitute", "T05", "ex",
     r"%s/,\s*/, /g", "a,b,  c,d", [1, 1],
     "Normalise the spacing after every comma to one space"),

    ("Count to four", "increment", "T08", "keys",
     "yyp<C-a>yyp<C-a>yyp<C-a>", "item 1", [1, 1],
     "Extend the numbered list to four items"),
    ("Continue the count", "insert-edit", "T08", "keys",
     "o2<Esc>o3<Esc>", "1", [1, 1],
     "Continue the count with two and three on their own lines"),
    ("Four copies of the line", "move-copy", "T08", "ex",
     "t.|t.|t.", "dup", [1, 1],
     "Make four copies of the line in total"),
    ("Three placeholder rows", "insert-edit", "T08", "keys",
     "3o-<Esc>", "list:", [1, 1],
     "Add three dash placeholder lines below"),

    ("Doubles to singles", "substitute", "T11", "ex",
     r"%s/\"\([^\"]*\)\"/'\1'/g", "say \"hi\" and \"bye\"", [1, 1],
     "Change the double quotes to single quotes"),
    ("Quote every word", "substitute", "T11", "ex",
     r"%s/\<\w\+\>/\"&\"/g", "ab cd ef", [1, 1],
     "Wrap every word in double quotes"),
    ("Parens to brackets", "substitute", "T11", "ex",
     r"%s/(\([^)]*\))/[\1]/", "call(arg one)", [1, 1],
     "Change the parentheses to square brackets"),

    ("Repoint the shebang", "substitute", "T12", "ex",
     r"1s@.*@#!/usr/bin/env python3@", "#!/usr/bin/python\nmain()", [1, 1],
     "Point the shebang at env python3"),
    ("Comment a config line", "substitute", "T12", "ex",
     "2s/^/# /", "keep = on\ndisable = maybe", [1, 1],
     "Comment out the second line with a hash"),

    ("Keep only the values", "substitute", "T14", "ex",
     r"%s/^.*: //", "name: jane\ncity: oslo", [1, 1],
     "Keep only the values after the colons"),
    ("Strip the markup", "substitute", "T14", "ex",
     r"%s/<[^>]*>//g", "<b>bold</b> and <i>italic</i>", [1, 1],
     "Remove all the HTML tags"),

    ("Undouble the words", "substitute", "T15", "ex",
     r"%s/\<\(\w\+\) \1\>/\1/g", "the the cat sat sat down", [1, 1],
     "Remove the accidentally doubled words"),
    ("Breathe after commas", "substitute", "T15", "ex",
     r"%s/,\(\S\)/, \1/g", "one,two,three words", [1, 1],
     "Put a space after each comma"),
    ("Start the sentence properly", "case-ops", "T15", "keys",
     "gUl", "start here", [1, 1],
     "Capitalise the first letter of the sentence"),

    ("Five of a kind", "repeat-count", "T16", "keys",
     "yy4p", "tick", [1, 1],
     "Make the line appear five times in total"),
    ("Duplicate in place", "move-copy", "T16", "ex",
     "t.", "once\nend", [1, 1],
     "Duplicate the first line"),
    ("Copy the middle to the end", "move-copy", "T16", "ex",
     "2,3t$", "head\naa\nbb\ntail", [1, 1],
     "Copy the two middle lines to the end of the file"),

    ("Lookbehind match", "substitute", "T17", "ex",
     r"%s/\(foo\)\@<=bar/X/g", "foobar bazbar foobar", [1, 1],
     "Replace bar with X only where foo comes before it"),
    ("Shortest match wins", "substitute", "T17", "ex",
     r"%s/\".\{-}\"/\"\"/", "\"first\" and \"second\"", [1, 1],
     "Empty the first quoted string, leaving the second alone"),
    ("Match end craft", "substitute", "T17", "ex",
     r"%s/\<cat\ze\w/CAT/g", "cat cats category", [1, 1],
     "Uppercase cat only where it begins a longer word"),

    ("Underline the title", "line-ops", "T18", "keys",
     "yyp:s/./=/g\n", "Chapter One", [1, 1],
     "Underline the title with a row of equals signs"),
    ("Left border", "substitute", "T18", "ex",
     r"%s/^/| /", "cell a\ncell b", [1, 1],
     "Add a left border bar to every line"),
    ("Right border", "substitute", "T18", "ex",
     r"%s/$/ |/", "cell a\ncell b", [1, 1],
     "Add a right border bar to every line"),
    ("Dashed rule", "substitute", "T18", "ex",
     r"%s/./-/g", "some header", [1, 1],
     "Turn the line into a dashed rule of the same length"),
    ("Box lid", "insert-edit", "T18", "keys",
     "O+----+<Esc>", "|text|", [1, 1],
     "Add a box top above the line"),

    ("Scramble a word", "operator-motion", "T19", "keys",
     "g?iw", "secret text", [1, 1],
     "Apply ROT13 to the first word"),
    ("Scramble the line", "operator-motion", "T19", "keys",
     "g?g?", "abc xyz", [1, 1],
     "Apply ROT13 to the whole line"),
    ("Mask the digits", "substitute", "T19", "ex",
     r"%s/\d/#/g", "card 1234 5678", [1, 1],
     "Mask every digit with a hash"),
    ("Blank the vowels", "substitute", "T19", "ex",
     r"%s/[aeiou]/_/g", "encode this text", [1, 1],
     "Replace every lowercase vowel with an underscore"),
    ("Flip the bits", "substitute", "T19", "ex",
     r"%s/[01]/\=1-submatch(0)/g", "1010 0110", [1, 1],
     "Invert every binary digit"),

    ("Markers, spelled out", "global", "T20", "ex",
     r"g/^<<<<<<<\|^=======\|^>>>>>>>/d",
     "intro\n<<<<<<< HEAD\nmine\n=======\nyours\n>>>>>>> main\noutro", [1, 1],
     "Drop the conflict markers and keep both sides"),
    ("Un-diff the removals", "substitute", "T20", "ex",
     r"%s/^-//", "-keep\n-these\n-lines", [1, 1],
     "Strip the leading diff minus signs"),
    ("Accept the additions", "substitute", "T20", "ex",
     r"%s/^+//", "+new line\ncontext\n+another", [1, 1],
     "Strip the plus prefixes from the added lines"),
    ("Drop the hunk headers", "global", "T20", "ex",
     r"g/^@@/d", "@@ -1,2 +1,2 @@\nbody\n@@ -9 +9 @@\nmore", [1, 1],
     "Delete the diff hunk header lines"),
]

THEMES = {
    "T01": "Case transformations", "T02": "Data-format conversion",
    "T03": "Code-to-code translation", "T04": "Micro-refactors",
    "T05": "Alignment & columns", "T06": "Sorting & dedup",
    "T07": "Numbering & counting", "T08": "Sequence generation",
    "T09": "Reordering & transposition", "T10": "Wrap, join & split",
    "T11": "Quoting & brackets", "T12": "Structured-config editing",
    "T13": "Whitespace repair", "T14": "Extraction & filtering",
    "T15": "Prose & typo repair", "T16": "Duplication-with-variation",
    "T17": "Regex-craft", "T18": "ASCII art", "T19": "Encodings",
    "T20": "Merge/diff repair",
}


def build_seeds():
    sys.path.insert(0, str(REPO / "scripts"))
    from make_v3_emulator import guess_tier
    out = []
    for (label, cat, theme, mode, cmd, buf, cur, hint) in SEEDS:
        res = run_vim(buf, cmd, mode=mode, cursor=tuple(cur))
        if not res.get("was_valid") or res.get("blocked"):
            sys.exit("seed %r no longer executes: %s"
                     % (label, res.get("blocked") or res.get("stderr", "")[:100]))
        after = res["buffer_after"]
        if after.rstrip("\n") == buf.rstrip("\n"):
            sys.exit("seed %r is a no-op" % label)
        out.append({"label": label, "category": cat,
                    "tier": guess_tier(mode, cmd),
                    "theme": theme, "mode": mode, "command": cmd,
                    "buffer_before": buf, "cursor": cur, "after": after,
                    "hint": hint})
    return out


def build_targets():
    rows = []
    for ds in ("v3", "v4"):
        p = REPO / "data" / ds / ("%s_cases.jsonl" % ds)
        if p.exists():
            rows += [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    tax = yaml.safe_load((REPO / "data/taxonomy.yaml").read_text(encoding="utf-8"))
    cat_tot = collections.Counter(r.get("category") for r in rows)
    th_tot = collections.Counter(r.get("theme") for r in rows if r.get("theme"))
    cats = []
    for c, v in tax["categories"].items():
        quota = sum(v["quotas"].values())
        cats.append({"key": c, "have": cat_tot.get(c, 0), "quota": quota})
    cats.sort(key=lambda x: x["have"] / x["quota"])
    themes = [{"key": t, "name": n, "have": th_tot.get(t, 0)}
              for t, n in THEMES.items()]
    return {"cats": cats, "themes": themes, "total": len(rows)}


TEMPLATE = r"""<title>Case Forge</title>
<style>
:root{
  --ground:#F2F0EE; --surface:#FCFBFA; --sunk:#E9E5E2; --raise:#E4DFDB;
  --ink:#231E22; --ink2:#4A4046; --muted:#6E6369; --faint:#9A8F95;
  --line:#DAD3CF; --line2:#E7E1DD;
  --accent:#6B2D5C; --accent2:#8A3E77; --accent-soft:#EFE2EC; --on-accent:#FCFBFA;
  --pass:#2E6E4E; --pass-bg:#DFEDE5; --pass-line:#AACFBB;
  --warn:#8C5A15; --warn-bg:#F4E9D8; --warn-line:#DFC79A;
  --stop:#9C3535; --stop-bg:#F3E1E0; --stop-line:#DFB4B2;
  --mono:ui-monospace,"SF Mono","Cascadia Mono","DejaVu Sans Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --disp:"Charter","Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
}
:root:not([data-theme="light"]){ color-scheme:light dark; }
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#151215; --surface:#1D191C; --sunk:#120F12; --raise:#282227;
    --ink:#E7E0E4; --ink2:#C0B6BC; --muted:#8D8288; --faint:#665D63;
    --line:#332C31; --line2:#282227;
    --accent:#C98FBA; --accent2:#DDA9CF; --accent-soft:#2B1F28; --on-accent:#1D191C;
    --pass:#6BC195; --pass-bg:#17281F; --pass-line:#2C4736;
    --warn:#D6A85C; --warn-bg:#2A2216; --warn-line:#493B20;
    --stop:#E08484; --stop-bg:#2C1B1B; --stop-line:#4A2A2A;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --ground:#151215; --surface:#1D191C; --sunk:#120F12; --raise:#282227;
  --ink:#E7E0E4; --ink2:#C0B6BC; --muted:#8D8288; --faint:#665D63;
  --line:#332C31; --line2:#282227;
  --accent:#C98FBA; --accent2:#DDA9CF; --accent-soft:#2B1F28; --on-accent:#1D191C;
  --pass:#6BC195; --pass-bg:#17281F; --pass-line:#2C4736;
  --warn:#D6A85C; --warn-bg:#2A2216; --warn-line:#493B20;
  --stop:#E08484; --stop-bg:#2C1B1B; --stop-line:#4A2A2A;
}
*{ box-sizing:border-box; }
body{ margin:0; background:var(--ground); color:var(--ink); font-family:var(--sans);
  font-size:15px; line-height:1.5; -webkit-font-smoothing:antialiased; }
h1,h2,h3{ font-family:var(--disp); font-weight:600; margin:0; letter-spacing:-.01em; }
.mono{ font-family:var(--mono); }
.lab{ font-family:var(--mono); font-size:10px; letter-spacing:.13em; text-transform:uppercase;
  color:var(--muted); font-weight:600; }
:focus-visible{ outline:2px solid var(--accent); outline-offset:2px; border-radius:3px; }
.wrap{ max-width:1240px; margin:0 auto; padding:0 20px; }
header{ background:var(--surface); border-bottom:1px solid var(--line); }
header .wrap{ padding:20px 20px 14px; }
h1{ font-size:clamp(24px,3.4vw,32px); margin:6px 0 8px; }
.lede{ max-width:72ch; color:var(--ink2); font-size:14.5px; }
.lede b{ color:var(--ink); }
.targets{ display:flex; flex-direction:column; gap:8px; margin-top:12px; }
.tgroup .lab{ display:block; margin-bottom:4px; }
.tchips{ display:flex; flex-wrap:wrap; gap:5px; }
.tchip{ font-family:var(--mono); font-size:11px; padding:3px 8px; border-radius:4px;
  cursor:pointer; background:var(--surface); border:1px solid var(--line); color:var(--ink2); }
.tchip:hover{ border-color:var(--accent); color:var(--accent); }
.tchip b{ font-weight:700; }
.tchip.hot{ background:var(--stop-bg); border-color:var(--stop-line); color:var(--stop); }
.tchip.warm{ background:var(--warn-bg); border-color:var(--warn-line); color:var(--warn); }
.tchip[aria-pressed="true"]{ outline:2px solid var(--accent); }

main.wrap{ padding:18px 20px 150px; }
.cols{ display:grid; grid-template-columns:minmax(430px,3fr) minmax(320px,2fr); gap:16px; }
@media (max-width:960px){ .cols{ grid-template-columns:1fr; } }
.panel{ background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:15px 16px; }
.panel h2{ font-size:18px; margin-bottom:2px; }
.panel p.sub{ margin:0 0 11px; color:var(--muted); font-size:12.5px; }
.field{ margin-top:10px; }
.field label{ display:block; margin-bottom:3px; }
.labnote{ text-transform:none; letter-spacing:0; font-weight:400; color:var(--faint); }
.fieldnote{ margin:5px 0 0; font-size:11.5px; color:var(--muted); line-height:1.45; }
.fieldnote b{ color:var(--warn); }
input[type=text],textarea,select,input[type=number]{
  width:100%; font:inherit; font-size:13.5px; padding:6px 8px; border:1px solid var(--line);
  border-radius:5px; background:var(--ground); color:var(--ink); }
textarea{ font-family:var(--mono); font-size:12.5px; resize:vertical; min-height:3.2em; }
textarea.cmd{ min-height:2.4em; font-weight:600; }
input:focus,textarea:focus,select:focus{ outline:none; border-color:var(--accent);
  box-shadow:0 0 0 2px var(--accent-soft); }
.row{ display:flex; gap:8px; }
.row>.field{ flex:1 1 auto; margin-top:10px; }
.row>.field.small{ flex:0 0 92px; }
.checks{ display:flex; flex-direction:column; gap:4px; margin-top:12px; }
.chk{ font-family:var(--mono); font-size:11.5px; padding:3px 8px; border-radius:4px; border:1px solid; }
.chk.ok{ background:var(--pass-bg); color:var(--pass); border-color:var(--pass-line); }
.chk.warn{ background:var(--warn-bg); color:var(--warn); border-color:var(--warn-line); }
.chk.err{ background:var(--stop-bg); color:var(--stop); border-color:var(--stop-line); font-weight:600; }
.btn{ font-family:var(--mono); font-size:12px; padding:6px 12px; border-radius:5px; cursor:pointer;
  background:var(--surface); border:1px solid var(--line); color:var(--ink2); }
.btn:hover{ border-color:var(--accent); color:var(--accent); }
.btn.pri{ background:var(--accent); border-color:var(--accent); color:var(--on-accent); font-weight:600; }
.btn.pri:hover{ background:var(--accent2); border-color:var(--accent2); }
.btn:disabled{ opacity:.45; cursor:not-allowed; }
.actions{ display:flex; gap:8px; margin-top:13px; flex-wrap:wrap; }

.seeds{ margin-top:15px; }
.seedgrid{ display:grid; grid-template-columns:repeat(auto-fill,minmax(215px,1fr)); gap:8px; margin-top:9px; }
.seed{ text-align:left; background:var(--surface); border:1px solid var(--line); border-radius:6px;
  padding:9px 11px; cursor:pointer; font:inherit; color:inherit; }
.seed:hover{ border-color:var(--accent); }
.seed .nm{ font-weight:600; font-size:13px; }
.seed .cm{ font-family:var(--mono); font-size:10.5px; color:var(--accent);
  overflow-wrap:anywhere; margin-top:3px; display:block; }
.seed .tg{ display:flex; gap:5px; flex-wrap:wrap; margin-top:5px; }
.seed .tg i{ font-style:normal; font-family:var(--mono); font-size:9px; padding:1px 5px;
  border-radius:3px; border:1px solid; }
.seed .tg .tv{ background:var(--accent-soft); color:var(--accent); border-color:var(--accent); }
.seed .tg .tgo{ background:var(--warn-bg); color:var(--warn); border-color:var(--warn-line); }
.seed .tg .tm{ background:var(--raise); color:var(--muted); border-color:var(--line); }

.drafts{ display:flex; flex-direction:column; gap:8px; margin-top:10px; }
.draft{ background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--pass);
  border-radius:6px; padding:9px 12px; }
.draft.err{ border-left-color:var(--stop); }
.draft .top{ display:flex; gap:8px; align-items:baseline; flex-wrap:wrap; }
.draft .ins{ font-weight:600; font-size:13.5px; flex:1 1 auto; }
.draft .cm{ font-family:var(--mono); font-size:11px; color:var(--accent); overflow-wrap:anywhere; }
.draft .meta{ font-family:var(--mono); font-size:10px; color:var(--muted); margin-top:3px; }
.draft .ops{ display:flex; gap:5px; margin-top:6px; }
.draft .ops button{ font-family:var(--mono); font-size:10.5px; padding:2px 8px; border-radius:3px;
  cursor:pointer; background:none; border:1px solid var(--line); color:var(--muted); }
.draft .ops button:hover{ border-color:var(--accent); color:var(--accent); }
.empty{ color:var(--muted); font-family:var(--mono); font-size:12px; padding:14px 0; }
.pill{ font-family:var(--mono); font-size:9.5px; font-weight:700; letter-spacing:.05em;
  text-transform:uppercase; padding:1px 6px; border-radius:3px; border:1px solid; white-space:nowrap; }
.p-ex{ background:var(--accent-soft); color:var(--accent); border-color:var(--accent); }
.p-keys{ background:var(--warn-bg); color:var(--warn); border-color:var(--warn-line); }

.dock{ position:fixed; left:0; right:0; bottom:0; z-index:30; background:var(--surface);
  border-top:1px solid var(--line); }
.dock .wrap{ display:flex; gap:10px; align-items:center; padding:9px 20px; flex-wrap:wrap; }
.dock .count{ font-family:var(--mono); font-size:12px; color:var(--muted); margin-left:auto; }
dialog{ max-height:92vh; overflow:auto; border:1px solid var(--line); border-radius:8px; background:var(--surface); color:var(--ink);
  max-width:min(880px,94vw); width:100%; padding:0; }
dialog::backdrop{ background:rgba(20,16,19,.55); }
dialog .dh{ padding:15px 18px 0; }
dialog .db{ padding:12px 18px 16px; display:flex; flex-direction:column; gap:9px; }
dialog textarea{ min-height:230px; font-size:11.5px; }
dialog .row2{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.cmdline{ font-family:var(--mono); font-size:11.5px; background:var(--sunk); border:1px solid var(--line);
  border-radius:4px; padding:7px 9px; color:var(--ink2); overflow-x:auto; white-space:pre; }
dialog p.sub{ margin:0; color:var(--muted); font-size:12.5px; }
.boot{ padding:30px 0; text-align:center; color:var(--muted); font-family:var(--mono); font-size:13px; }
.boot.err{ color:var(--stop); text-align:left; white-space:pre-wrap; overflow-wrap:anywhere;
  background:var(--stop-bg); border:1px solid var(--stop-line); border-radius:6px; padding:14px 16px; }
</style>

<header><div class="wrap">
  <div class="lab">vimBot &middot; dataset v3 &middot; hand authoring</div>
  <h1>Case Forge</h1>
  <p class="lede">
    Draft new cases against the corpus gaps. The page checks what a browser can check &mdash;
    the &sect;6 instruction rules, the leak guard, cursor bounds, duplicates &mdash; and keeps
    your drafts in this browser as you type. <b>Vim has the final word:</b> export and run the
    drafts through <span class="mono">ingest_handcrafted.py</span>, which executes each one in
    the sandbox and refuses anything invalid, no-op, or duplicated before it reaches v3.
    Click a target to aim the editor; click a starter to load a verified command.
  </p>
  <div class="targets" id="targets"></div>
</div></header>

<main class="wrap">
  <div class="boot" id="boot">Loading&hellip; if this message stays, the page failed to start
  and the reason will appear here.</div>
  <div class="cols" style="display:none" id="cols">
    <div>
      <div class="panel" id="editor">
        <h2 id="edtitle">New draft</h2>
        <p class="sub">The instruction describes <b>what the user wants</b>, never the keys.
        Ex commands are stored without the leading colon.</p>
        <div class="field"><label class="lab" for="f-ins">Instruction</label>
          <textarea id="f-ins" rows="2" placeholder="Wrap every line in double quotes"></textarea></div>
        <div class="row">
          <div class="field small"><label class="lab" for="f-mode">Mode</label>
            <select id="f-mode"><option value="ex">ex</option><option value="keys">keys</option></select></div>
          <div class="field"><label class="lab" for="f-cmd">Command</label>
            <textarea id="f-cmd" class="cmd" rows="1" spellcheck="false"></textarea></div>
        </div>
        <div class="field"><label class="lab" for="f-buf">Buffer before
          <span class="labnote">Tab types a real tab here</span></label>
          <textarea id="f-buf" rows="4" spellcheck="false" placeholder="one line per line, literally"></textarea></div>
        <div class="row">
          <div class="field small"><label class="lab" for="f-cl">Cursor line</label>
            <input type="number" id="f-cl" value="1" min="1"></div>
          <div class="field small"><label class="lab" for="f-cc">Cursor col</label>
            <input type="number" id="f-cc" value="1" min="1"></div>
          <div class="field"><label class="lab" for="f-cat">Vim category</label>
            <select id="f-cat"></select></div>
          <div class="field small"><label class="lab" for="f-theme">Golf theme</label>
            <select id="f-theme"></select></div>
        </div>
        <div class="field"><label class="lab" for="f-exp">Expected result
          <span class="labnote">optional &mdash; if given, vim must agree at ingest or the draft is refused. Tab types a real tab here</span></label>
          <textarea id="f-exp" rows="2" spellcheck="false" placeholder="leave empty to accept whatever vim produces"></textarea>
          <p class="fieldnote"><b>Indentation is tabs, not spaces:</b> the sandbox runs vim's bare
          defaults (shiftwidth 8, noexpandtab), so <span class="mono">&gt;&gt;</span> and friends
          insert a literal tab character. Expect <span class="mono">\t</span>-indented output, or
          leave this field empty and let vim's answer stand.</p></div>
        <div class="checks" id="checks"></div>
        <div class="actions">
          <button class="btn pri" id="save">Add draft</button>
          <button class="btn" id="clear">Clear</button>
        </div>
      </div>
      <div class="seeds">
        <span class="lab" id="seedlab">Starters</span>
        <div class="seedgrid" id="seeds"></div>
      </div>
    </div>
    <div>
      <div class="panel">
        <h2>Drafts</h2>
        <p class="sub">Kept in this browser. Export from the bar below when you have a batch.</p>
        <div class="drafts" id="drafts"></div>
      </div>
    </div>
  </div>
</main>

<div class="dock"><div class="wrap">
  <button class="btn pri" id="export">Export batch</button>
  <button class="btn" id="import">Import / restore</button>
  <span class="count" id="count"></span>
</div></div>

<dialog id="dlg"><div class="dh"><h2 id="dlgtitle">Export</h2></div><div class="db">
  <div class="row2">
    <button class="btn" id="copy">Copy JSON</button>
    <button class="btn" id="load" style="display:none">Load into drafts</button>
    <span style="flex:1"></span>
    <button class="btn" id="close">Close</button>
  </div>
  <textarea id="out" spellcheck="false"></textarea>
  <div class="cmdline" id="cmd2">conda run -n vimBot python scripts/ingest_handcrafted.py \
  --input drafts.json --dataset v4 --provenance owner-authored --dry-run</div>
  <p class="sub">Save the JSON as <span class="mono">drafts.json</span>, dry-run the command
  above, fix anything it rejects, then drop <span class="mono">--dry-run</span>. Afterwards
  rebuild the corpus (<span class="mono">build_v3_corpus.py</span>) and re-run the gold
  self-check. Nothing enters v3 without executing in vim first.</p>
</div></dialog>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const P = JSON.parse(document.getElementById("payload").textContent);
const KEY = "vimbot.case-forge.v1";
let drafts = [];
try { drafts = JSON.parse(localStorage.getItem(KEY) || "[]"); } catch (e) { drafts = []; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(drafts)); } catch (e) {} };
let editing = null;   // index into drafts, or null for new

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ---- tier heuristic, ported from make_v3_emulator.guess_tier — display
   only; the ingest script recomputes it with the Python original ---- */
function guessTier(cmd) {
  cmd = cmd || "";
  if (/\\\(|\\\||\\%|\\z[se]|\\[dwsa]|\[\^?.+\]|\\=/.test(cmd)) return "T3";
  const ops = (cmd.match(/[dcyxsprJ~<>]|g[uU~q?]|<C-[ax]>/g) || []).length;
  if (ops >= 3) return "T3";
  if (/\d/.test(cmd) || /\/[gice]+\s*$|!\s*$|\s+[niur]\b/.test(cmd) || ops === 2) return "T2";
  return "T1";
}

/* ---- leak guard, mirroring apply_instructions.leaks (advisory; the ingest
   script re-runs the real one) ---- */
function leakForms(cmd) {
  const forms = new Set();
  const f = (cmd || "").trim();
  if (f) {
    forms.add(f);
    const stripped = f.replace(/<[^>]*>/g, "");
    forms.add(stripped);
    forms.add(stripped.replace(/^:+/, ""));
  }
  return [...forms].map(x => x.trim().toLowerCase()).filter(x => x.length >= 2);
}
function checkDraft(d, self) {
  if (self === undefined) self = drafts.indexOf(d);
  const out = [];
  const ins = (d.instruction || "").trim();
  if (!ins) out.push(["err", "instruction is empty"]);
  else {
    const low = ins.toLowerCase();
    for (const f of leakForms(d.command)) {
      if (low.includes(f)) { out.push(["err", "leak: instruction contains " + JSON.stringify(f)]); break; }
    }
    if (/\bpress\b|\bkeystroke|\btype the key/.test(low))
      out.push(["err", "leak: describes keystrokes, not intent"]);
    if (ins.endsWith(".")) out.push(["warn", "trailing period"]);
    if (/\n/.test(ins)) out.push(["warn", "more than one line"]);
    if (ins[0] !== ins[0].toUpperCase()) out.push(["warn", "not sentence case"]);
    if (ins.length < 12) out.push(["warn", "very short"]);
  }
  if (!(d.command || "").trim()) out.push(["err", "command is empty"]);
  if (d.mode === "ex" && /<[a-z-]+>/i.test(d.command || ""))
    out.push(["warn", "angle-bracket keys in an ex command — is the mode right?"]);
  const buf = d.buffer_before || "";
  if (!buf) out.push(["warn", "empty buffer (allowed, but rare)"]);
  const lines = buf.split("\n");
  if (lines.length && lines[lines.length - 1] === "") lines.pop();
  const L = d.cursor[0], C = d.cursor[1];
  const nl = Math.max(1, lines.length);
  if (L < 1 || L > nl) out.push(["err", "cursor line " + L + " outside the " + nl + "-line buffer"]);
  else {
    const len = (lines[L - 1] || "").length;
    if (C < 1 || C > len + 1) out.push(["err", "cursor col " + C + " outside line " + L + " (" + len + " chars)"]);
  }
  drafts.forEach((o, i) => {
    if (i !== self && o.instruction.trim().toLowerCase() === ins.toLowerCase()
        && o.buffer_before === buf)
      out.push(["err", "duplicate of draft " + (i + 1)]);
  });
  return out;
}

/* ---- editor ---- */
function readForm() {
  return {
    instruction: $("f-ins").value,
    mode: $("f-mode").value,
    command: $("f-mode").value === "ex" ? $("f-cmd").value.replace(/^:+/, "") : $("f-cmd").value,
    buffer_before: $("f-buf").value,
    cursor: [parseInt($("f-cl").value || "1", 10), parseInt($("f-cc").value || "1", 10)],
    category: $("f-cat").value,
    theme: $("f-theme").value,
    expected_after: $("f-exp").value || undefined,
  };
}
function fillForm(d) {
  $("f-ins").value = d.instruction || "";
  $("f-mode").value = d.mode || "ex";
  $("f-cmd").value = d.command || "";
  $("f-buf").value = d.buffer_before || "";
  $("f-cl").value = (d.cursor && d.cursor[0]) || 1;
  $("f-cc").value = (d.cursor && d.cursor[1]) || 1;
  $("f-cat").value = d.category || P.targets.cats[0].key;
  $("f-theme").value = d.theme || "";
  $("f-exp").value = d.expected_after || "";
  paintChecks();
}
function paintChecks() {
  const res = checkDraft(readForm(), editing === null ? -1 : editing);
  const tierChip = '<span class="chk ok">tier ' + guessTier(readForm().command) +
    ' — detected from the command, not typed</span>';
  $("checks").innerHTML = (res.length
    ? res.map(([k, m]) => '<span class="chk ' + (k === "err" ? "err" : "warn") + '">' + esc(m) + "</span>").join("")
    : '<span class="chk ok">passes every static check — vim gets the final word at ingest</span>')
    + tierChip;
  $("save").disabled = res.some(([k]) => k === "err");
}
["f-ins", "f-cmd", "f-buf", "f-cl", "f-cc", "f-mode", "f-exp"].forEach(id =>
  $(id).addEventListener("input", paintChecks));
["f-buf", "f-exp"].forEach(id => $(id).addEventListener("keydown", e => {
  if (e.key !== "Tab" || e.shiftKey || e.ctrlKey || e.altKey || e.metaKey) return;
  e.preventDefault();
  const t = e.currentTarget, a = t.selectionStart, b = t.selectionEnd;
  t.value = t.value.slice(0, a) + "\t" + t.value.slice(b);
  t.selectionStart = t.selectionEnd = a + 1;
  t.dispatchEvent(new Event("input"));
}));
$("f-cmd").addEventListener("input", () => {
  const v = $("f-cmd").value;
  if (/^:/.test(v)) $("f-mode").value = "ex";
  else if (/<(esc|cr|bs|tab|space|c-.)>/i.test(v)) $("f-mode").value = "keys";
  paintChecks();
});

$("save").addEventListener("click", () => {
  const d = readForm();
  if (checkDraft(d, editing === null ? -1 : editing).some(([k]) => k === "err")) return;
  if (editing === null) drafts.push(d); else drafts[editing] = d;
  editing = null; $("edtitle").textContent = "New draft";
  save(); paintDrafts(); fillForm({ category: d.category, theme: d.theme, mode: d.mode });
});
$("clear").addEventListener("click", () => {
  editing = null; $("edtitle").textContent = "New draft"; fillForm({});
});

/* ---- targets ---- */
let galleryFilter = null;   // {cat} or {theme} or null = all
function paintTargets() {
  const t = P.targets;
  const cats = t.cats.map(c => {
    const frac = c.have / c.quota;
    const cls = frac < 0.1 ? "hot" : (frac < 0.5 ? "warm" : "");
    return '<button class="tchip ' + cls + '" data-cat="' + esc(c.key) + '"><b>' +
      esc(c.key) + "</b> " + c.have + "/" + c.quota + "</button>";
  }).join("");
  const themes = t.themes.map(x => {
    const cls = x.have === 0 ? "hot" : (x.have < 20 ? "warm" : "");
    return '<button class="tchip ' + cls + '" data-theme="' + esc(x.key) +
      '" data-cat="substitute"><b>' + esc(x.key) + "</b> " + esc(x.name) +
      " · " + x.have + "</button>";
  }).join("");
  $("targets").innerHTML =
    '<div class="tgroup"><span class="lab">vim taxonomy — rows / quota</span><div class="tchips">' +
    cats + '</div></div>' +
    '<div class="tgroup"><span class="lab">vimgolf themes — rows tagged</span><div class="tchips">' +
    themes + '</div></div>';
  $("targets").querySelectorAll("[data-cat]").forEach(b => b.addEventListener("click", () => {
    const again = b.getAttribute("aria-pressed") === "true";
    $("targets").querySelectorAll(".tchip").forEach(x =>
      x.setAttribute("aria-pressed", !again && x === b));
    if (again) { galleryFilter = null; }
    else {
      $("f-cat").value = b.dataset.cat;
      if (b.dataset.theme) { $("f-theme").value = b.dataset.theme; galleryFilter = { theme: b.dataset.theme }; }
      else { galleryFilter = { cat: b.dataset.cat }; }
    }
    paintChecks(); paintSeeds();
    document.querySelector(".seeds").scrollIntoView({ block: "nearest", behavior: "smooth" });
  }));
}

/* ---- seeds ---- */
function paintSeeds() {
  let shown = P.seeds.map((s, i) => [s, i]);
  if (galleryFilter && galleryFilter.cat)
    shown = shown.filter(([s]) => s.category === galleryFilter.cat);
  if (galleryFilter && galleryFilter.theme)
    shown = shown.filter(([s]) => s.theme === galleryFilter.theme);
  $("seedlab").textContent = "Starters — " +
    (galleryFilter
      ? (galleryFilter.cat ? "vim category " + galleryFilter.cat
                           : "vimgolf theme " + galleryFilter.theme) +
        " (" + shown.length + ")"
      : "all (" + shown.length + ")") +
    " — every command verified through the sandbox at build time";
  if (!shown.length) {
    $("seeds").innerHTML = '<div class="empty">No starters here yet — this gap is all yours. ' +
      'The editor is aimed at it; write from scratch.</div>';
    return;
  }
  $("seeds").innerHTML = shown.map(([s, i]) =>
    '<button class="seed" data-seed="' + i + '"><span class="nm">' + esc(s.label) + "</span>" +
    '<span class="cm">' + esc(s.mode === "ex" ? ":" + s.command : s.command) + "</span>" +
    '<span class="tg"><i class="tv">vim ' + esc(s.category) + "</i>" +
    (s.theme ? '<i class="tgo">golf ' + esc(s.theme) + "</i>" : "") +
    '<i class="tm">' + esc(s.mode) + " · " + esc(s.tier) + "</i></span></button>").join("");

  $("seeds").querySelectorAll("[data-seed]").forEach(b => b.addEventListener("click", () => {
    const s = P.seeds[+b.dataset.seed];
    editing = null; $("edtitle").textContent = "New draft (from starter: " + s.label + ")";
    fillForm({ instruction: s.hint, mode: s.mode, command: s.command,
               buffer_before: s.buffer_before, cursor: s.cursor,
               category: s.category, theme: s.theme,
               expected_after: s.after.replace(/\n$/, "") });
    $("f-ins").focus(); $("f-ins").select();
  }));
}

/* ---- draft list ---- */
function paintDrafts() {
  const host = $("drafts");
  if (!drafts.length) {
    host.innerHTML = '<div class="empty">Nothing yet. Load a starter or write from scratch.</div>';
  } else {
    host.innerHTML = drafts.map((d, i) => {
      const bad = checkDraft(d).some(([k]) => k === "err");
      return '<div class="draft' + (bad ? " err" : "") + '">' +
        '<div class="top"><span class="pill p-' + d.mode + '">' + d.mode + "</span>" +
        '<span class="ins">' + esc(d.instruction || "(no instruction)") + "</span></div>" +
        '<span class="cm">' + esc(d.mode === "ex" ? ":" + d.command : d.command) + "</span>" +
        '<div class="meta">' + esc(d.category) + " · " + guessTier(d.command) + " auto" +
        (d.theme ? " · " + esc(d.theme) : "") + " · " +
        (d.buffer_before.split("\n").length) + "-line buffer · cursor " +
        d.cursor[0] + ":" + d.cursor[1] + "</div>" +
        '<div class="ops"><button data-e="' + i + '">edit</button>' +
        '<button data-d="' + i + '">duplicate</button>' +
        '<button data-x="' + i + '">delete</button></div></div>';
    }).join("");
  }
  host.querySelectorAll("[data-e]").forEach(b => b.addEventListener("click", () => {
    editing = +b.dataset.e; $("edtitle").textContent = "Editing draft " + (editing + 1);
    fillForm(drafts[editing]); window.scrollTo({ top: 0, behavior: "smooth" });
  }));
  host.querySelectorAll("[data-d]").forEach(b => b.addEventListener("click", () => {
    const c = JSON.parse(JSON.stringify(drafts[+b.dataset.d]));
    c.instruction = ""; drafts.push(c); save(); paintDrafts();
  }));
  host.querySelectorAll("[data-x]").forEach(b => b.addEventListener("click", () => {
    drafts.splice(+b.dataset.x, 1); if (editing !== null) { editing = null; $("edtitle").textContent = "New draft"; }
    save(); paintDrafts();
  }));
  const bad = drafts.filter(d => checkDraft(d).some(([k]) => k === "err")).length;
  $("count").textContent = drafts.length + " draft" + (drafts.length === 1 ? "" : "s") +
    (bad ? " · " + bad + " with problems (excluded from export)" : "");
}

/* ---- export / import ---- */
$("export").addEventListener("click", () => {
  const good = drafts.filter(d => !checkDraft(d).some(([k]) => k === "err"))
    .map(d => { const c = { ...d }; if (!c.expected_after) delete c.expected_after; return c; });
  $("dlgtitle").textContent = "Export " + good.length + " draft" + (good.length === 1 ? "" : "s");
  $("out").value = JSON.stringify(good, null, 1);
  $("out").readOnly = true; $("load").style.display = "none"; $("copy").style.display = "";
  $("cmd2").style.display = ""; $("dlg").showModal();
});
$("import").addEventListener("click", () => {
  $("dlgtitle").textContent = "Import drafts (paste a previous export)";
  $("out").value = ""; $("out").readOnly = false;
  $("load").style.display = ""; $("copy").style.display = "none";
  $("cmd2").style.display = "none"; $("dlg").showModal();
});
$("load").addEventListener("click", () => {
  let arr;
  try { arr = JSON.parse($("out").value); } catch (e) { alert("Not valid JSON: " + e.message); return; }
  if (!Array.isArray(arr)) { alert("Expected a JSON list of drafts."); return; }
  arr.forEach(d => drafts.push({
    instruction: d.instruction || "", mode: d.mode === "keys" ? "keys" : "ex",
    command: d.command || "", buffer_before: d.buffer_before || "",
    cursor: Array.isArray(d.cursor) ? [d.cursor[0] || 1, d.cursor[1] || 1] : [1, 1],
    category: d.category || "substitute",
    theme: d.theme || "", expected_after: d.expected_after,
  }));
  save(); paintDrafts(); $("dlg").close();
});
$("close").addEventListener("click", () => $("dlg").close());
$("copy").addEventListener("click", async e => {
  $("out").select();
  let ok = false;
  try { await navigator.clipboard.writeText($("out").value); ok = true; } catch (err) {}
  e.currentTarget.textContent = ok ? "Copied" : "Press Ctrl+C";
  setTimeout(() => { e.currentTarget.textContent = "Copy JSON"; }, 1900);
});

/* ---- boot ---- */
function boot() {
  $("f-cat").innerHTML = P.targets.cats.map(c =>
    '<option value="' + esc(c.key) + '">' + esc(c.key) + "</option>").join("") +
    P.extra_cats.map(c => '<option value="' + esc(c) + '">' + esc(c) + " (outside taxonomy)</option>").join("");
  $("f-theme").innerHTML = '<option value="">—</option>' + P.targets.themes.map(t =>
    '<option value="' + esc(t.key) + '">' + esc(t.key) + " " + esc(t.name) + "</option>").join("");
  paintTargets(); paintSeeds(); paintDrafts(); fillForm({});
  $("cols").style.display = ""; $("boot").remove();
}
try { boot(); } catch (err) {
  const b = $("boot");
  if (b) { b.className = "boot err";
    b.textContent = "This page failed to start.\n\n" + (err && err.stack ? err.stack : String(err)); }
  throw err;
}
</script>
"""

REQUIRED_JS = [r"if (/\n/.test(ins))", r"/\bpress\b|\bkeystroke|\btype the key/"]
FORBIDDEN = {chr(c) for c in list(range(0, 9)) + [11, 12] + list(range(14, 32))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = {
        "seeds": build_seeds(),
        "targets": build_targets(),
        "extra_cats": ["global", "visual", "visual-block", "misc-ex"],
    }
    blob = json.dumps(payload, ensure_ascii=False,
                      separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.replace("__PAYLOAD__", blob)

    print("  guard: " + check_page(page, REQUIRED_JS, "Case Forge"))
    out = pathlib.Path(args.out)
    out.write_text(page, encoding="utf-8")
    print("seeds: %d (all executed)  page: %.0f KB  -> %s"
          % (len(payload["seeds"]), len(page.encode()) / 1024, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
