CSS = """
<style>
.block-container {padding-top: 1.6rem;}
.app-title {font-size: 1.7rem; font-weight: 800; letter-spacing: .02em; margin-bottom: .1rem;}
.app-sub {color: #5b6472; margin-bottom: .8rem;}
.banner {border-left: 5px solid #d9822b; background: rgba(217,130,43,.10); padding: .6rem .9rem;
         border-radius: 6px; margin: .4rem 0 1rem 0; font-size: .92rem;}
.banner-sim {border-left-color: #7b5bd6; background: rgba(123,91,214,.10);}
div[data-testid="stMetric"] {background: rgba(120,130,150,.08); border-radius: 10px; padding: .6rem .8rem;}
</style>
"""

REVIEW_BANNER = ("<div class='banner'><b>Potential anomalies requiring human review.</b> "
                 "Alerts are rule-based candidates, not proof of a violation or collision. "
                 "Prototype only - not a certified road-safety system.</div>")

SIM_BANNER = ("<div class='banner banner-sim'><b>SIMULATED DATA.</b> This view shows the scripted demo scene. "
              "It is not AI inference and says nothing about detector accuracy.</div>")
