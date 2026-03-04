"""
[sdk/exceptions). — customer-facing exceptions for [t-bound).

DAYANCH: Define all exceptions here. Import from here everywhere.
Error messages must be clear and actionable for customers.

WHAT TO IMPLEMENT:

class TBoundError(Exception):
    '''Base exception for all [t-bound) errors.'''

class TBoundAuthError(TBoundError):
    '''Invalid or expired API key.'''
    # Message: "Invalid API key. Get yours at tbound.ai/dashboard"

class TBoundProjectError(TBoundError):
    '''Project not found or access denied.'''
    # Message: "Project '{project_id}' not found. Check your project name."

class TBoundInsufficientRuns(TBoundError):
    '''Not enough runs to make a recommendation.'''
    # Message: "Need at least 3 runs to recommend. You have {n}. Keep training."

class TBoundAPIError(TBoundError):
    '''Server error. Run has been buffered locally.'''
    # Message: "API unreachable. Run buffered — will retry automatically."

class TBoundConfigError(TBoundError):
    '''tbound.init() not called or misconfigured.'''
    # Message: "Call tbound.init() before tbound.log() or tbound.recommend()"
"""

# TODO: implement this file
raise NotImplementedError("sdk/exceptions.py not yet implemented — see docstring")
