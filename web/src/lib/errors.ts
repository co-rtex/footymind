import { ApiError } from "../api/client";
import { ContractError } from "../api/parse";

/** Plain-language explanation of a failed request, with the technical text kept separately. */
export interface ErrorView {
  /** what happened, in words a football viewer understands */
  message: string;
  /** the raw technical detail (status and server text), shown small for bug reports */
  technical: string;
  /** whether retrying the same request can help */
  retryable: boolean;
  /** the thing asked for does not exist (a stale or mistyped link) */
  notFound: boolean;
}

export function describeError(e: Error, subject = "this item"): ErrorView {
  const technical = e.message;
  if (e instanceof ApiError) {
    if (e.timedOut) {
      return {
        message: "The FootyMind server did not answer in time (timed out). It may be busy or stuck; try again.",
        technical,
        retryable: true,
        notFound: false,
      };
    }
    if (e.unreachable || e.status === 0) {
      return {
        message: "The FootyMind server is not responding. Check that it is running, then try again.",
        technical,
        retryable: true,
        notFound: false,
      };
    }
    if (e.status === 404) {
      return {
        message: `We could not find ${subject}. The link may be mistyped, or it may come from an older version of the processed data.`,
        technical,
        retryable: false,
        notFound: true,
      };
    }
    if (e.status === 422 || e.status === 400) {
      return { message: `The server could not use this request: ${e.detail}`, technical, retryable: false, notFound: false };
    }
    if (e.status >= 500) {
      return {
        message: "The server hit an error while computing this. Trying again may help.",
        technical,
        retryable: true,
        notFound: false,
      };
    }
  }
  if (e instanceof ContractError) {
    return {
      message:
        "The server answered with data this version of the app does not understand. Reload the page; if it persists, the app and server versions differ.",
      technical,
      retryable: true,
      notFound: false,
    };
  }
  return { message: technical, technical: "", retryable: true, notFound: false };
}
