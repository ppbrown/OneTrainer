# README-pyside.md 

Tips for developing with the pyside GUI library

1. In certain situations like mouse event handlers, pyside hides stackdumps, so you wont get the normal
file.py:linenumber
output to tell you exactly where the error is happening.

If you need to pull them out again the following may be useful

    import traceback

    def eventHandler(self, event):
        try:
            # your event code here
        except Exception:
            traceback.print_exc()
            raise

2. If you are wondering, "Is there a way to do X, with pyside?"
Just ask ChatGPT. Its usually pretty good at knowing alternatives.