"""SIMULATED payment / inventory / shipping services.

Nothing here talks to a real provider. Every response carries `"simulated": true` and an
`X-Relay-Simulated: true` header. The simulator has its own database and its own models,
mirroring the real-world boundary: Relay can never join across these tables, only call HTTP APIs.
"""
