# The A1 interface (paraphrased summary)

A1 is the interface between the **Non-RT RIC** and the **Near-RT RIC**. It
carries two kinds of information:

- **Policy (PM)** — the Non-RT RIC tells the Near-RT RIC what to optimise for
  and within which constraints, tagged with a policy type, a scope and a
  tenant label.
- **Enrichment (RNS)** — context that helps the Near-RT RIC and its xApps make
  better decisions, such as UE route information.

A1 is **not** a raw radio control channel. Control of the RAN elements happens
through E2 (Near-RT RIC to Du/Cu) and O1 (SMO to network functions). An agent
that claims to "apply control via A1" has confused policy intent with device
command, which is precisely the class of mistake a scope-enforcing sandbox
exists to catch.

Policy types are grouped (for example RAN function, QoE, load balancing), and
each policy instance is versioned so a Near-RT RIC can reject stale or
out-of-scope directives.

*Paraphrase for a demo corpus, not a specification excerpt.*
