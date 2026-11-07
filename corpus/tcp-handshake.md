# The TCP Three-Way Handshake

Before two computers exchange data using the Transmission Control Protocol, they establish a connection through a three-step exchange called the three-way handshake. The handshake synchronizes both sides on starting sequence numbers and confirms that each side is ready to send and receive.

In the first step, the initiating computer sends a segment with the SYN flag set, carrying its initial sequence number. In the second step, the receiving computer replies with a segment that has both the SYN and ACK flags set: it acknowledges the initiator's sequence number and offers its own. In the third step, the initiator sends an ACK segment acknowledging the other side's sequence number. Only then does data transfer begin.

Sequence numbers matter because TCP guarantees reliable, ordered delivery over an unreliable network. Every byte of data is numbered, the receiver acknowledges the bytes it has received, and the sender retransmits anything that is not acknowledged in time. If segments arrive out of order, the sequence numbers let the receiver reassemble the stream correctly.

Closing a connection uses a similar exchange of FIN and ACK segments so both sides can finish sending. A connection that is opened but never completed, for example when a server is flooded with SYN segments whose handshakes are never finished, can exhaust server resources; this SYN flood pattern is a classic denial-of-service attack, countered by techniques such as SYN cookies.
