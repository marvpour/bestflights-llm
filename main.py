"""
requirements:
  pip install pydantic amadeus python-dateutil openai

env:
  export OPENAI_API_KEY=...
  export AMADEUS_CLIENT_ID=...
  export AMADEUS_CLIENT_SECRET=...
"""

from __future__ import annotations
import os, json, math, datetime as dt
from typing import List, Optional, Tuple
from dateutil.parser import parse as parse_dt
from pydantic import BaseModel, Field, validator

# ---------- Data models ----------
class TravelQuery(BaseModel):
    origin: str = Field(..., description="IATA code or city name for departure")
    destination: str = Field(..., description="IATA code or city name for arrival")
    depart_date: dt.date = Field(..., description="Departure date (YYYY-MM-DD)")
    return_date: Optional[dt.date] = Field(None, description="Return date if round trip")
    passengers: int = Field(1, ge=1, description="Number of adult passengers")
    cabin: str = Field("ECONOMY", description="ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST")

    @validator("depart_date", "return_date", pre=True)
    def _to_date(cls, v):
        if v is None: return v
        if isinstance(v, dt.date): return v
        return parse_dt(str(v)).date()

class FlightSlice(BaseModel):
    depart_at: dt.datetime
    arrive_at: dt.datetime
    marketing_carrier: str
    flight_number: str
    duration_minutes: int
    origin: str
    destination: str
    stops: int

class FlightOption(BaseModel):
    id: str
    price_total: float
    currency: str
    slices: List[FlightSlice]
    cabin: str

    @property
    def total_duration_minutes(self) -> int:
        return sum(s.duration_minutes for s in self.slices)

    @property
    def total_stops(self) -> int:
        return sum(s.stops for s in self.slices)

# ---------- LLM wrappers ----------
class LLM:
    """Minimal OpenAI wrapper; swap with your provider if needed."""
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        """
        Ask model to return JSON matching `schema` (loosely). We post-hoc validate with pydantic.
        """
        msg = [
            {"role":"system","content": system},
            {"role":"user","content": user},
        ]
        tool = {
            "type": "function",
            "function": {
                "name": "emit_structured",
                "description": "Return structured JSON matching the provided JSON schema.",
                "parameters": schema
            }
        }
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msg,
            tools=[tool],
            tool_choice={"type":"function","function":{"name":"emit_structured"}},
            temperature=0.2,
        )
        call = resp.choices[0].message.tool_calls[0]
        return json.loads(call.function.arguments)

    def rewrite(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.5,
        )
        return resp.choices[0].message.content.strip()

# ---------- Agent 1: NLU / query understanding ----------
class NLUAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def parse(self, user_text: str) -> TravelQuery:
        schema = {
            "type":"object",
            "properties":{
                "origin":{"type":"string","description":"Departure city or IATA code"},
                "destination":{"type":"string","description":"Arrival city or IATA code"},
                "depart_date":{"type":"string","description":"YYYY-MM-DD"},
                "return_date":{"type":["string","null"],"description":"YYYY-MM-DD or null"},
                "passengers":{"type":"integer","minimum":1,"default":1},
                "trip_length_days":{"type":["integer","null"],"minimum":1,"description":"If return_date not given, infer using this many days"}
            },
            "required":["origin","destination","depart_date"],
            "additionalProperties":False
        }
        sys = ("Extract structured flight search details. "
               "Prefer IATA codes if mentioned; otherwise keep city names. "
               "If user says 'number of days', put it in trip_length_days and leave return_date null.")
        raw = self.llm.complete_json(sys, user_text, schema)

        # Fill return_date if only trip_length_days is present
        depart = parse_dt(raw["depart_date"]).date()
        ret = raw.get("return_date")
        days = raw.get("trip_length_days")
        if not ret and days:
            ret = depart + dt.timedelta(days=int(days))

        tq = TravelQuery(
            origin=raw["origin"],
            destination=raw["destination"],
            depart_date=depart,
            return_date=(parse_dt(ret).date() if ret else None),
            passengers=int(raw.get("passengers") or 1),
        )
        return tq

# ---------- Agent 2: Flight API client (interface + Amadeus example) ----------
class FlightAPI:
    def search(self, q: TravelQuery) -> List[FlightOption]:
        raise NotImplementedError

class AmadeusFlightAPI(FlightAPI):
    """
    Uses amadeus.shopping.flight_offers_search. You must have credentials.
    Docs: https://developers.amadeus.com/
    """
    def __init__(self):
        from amadeus import Client, ResponseError
        self.ResponseError = ResponseError
        self.client = Client(
            client_id=os.environ["AMADEUS_CLIENT_ID"],
            client_secret=os.environ["AMADEUS_CLIENT_SECRET"],
        )

    def _city_or_iata(self, s: str) -> str:
        # Very simple passthrough; in production, map cities to IATA codes.
        return s.strip().upper()

    def search(self, q: TravelQuery) -> List[FlightOption]:
        from amadeus import ResponseError
        params = dict(
            originLocationCode=self._city_or_iata(q.origin),
            destinationLocationCode=self._city_or_iata(q.destination),
            departureDate=q.depart_date.isoformat(),
            adults=q.passengers,
            travelClass=q.cabin,
            currencyCode="USD",
            max=30,
        )
        if q.return_date:
            params["returnDate"] = q.return_date.isoformat()

        try:
            res = self.client.shopping.flight_offers_search.get(**params)
        except ResponseError as e:
            raise RuntimeError(f"Amadeus error: {e}")

        options: List[FlightOption] = []
        for i, offer in enumerate(res.data):
            price_total = float(offer["price"]["grandTotal"])
            currency = offer["price"]["currency"]

            slices: List[FlightSlice] = []
            for itin in offer["itineraries"]:
                segs = itin["segments"]
                stops = max(0, len(segs) - 1)
                first = segs[0]
                last = segs[-1]
                duration_iso = itin["duration"]  # e.g., "PT11H35M"
                duration_minutes = _iso8601_duration_to_minutes(duration_iso)

                slices.append(FlightSlice(
                    depart_at=parse_dt(first["departure"]["at"]),
                    arrive_at=parse_dt(last["arrival"]["at"]),
                    marketing_carrier=first["carrierCode"],
                    flight_number=first["number"],
                    duration_minutes=duration_minutes,
                    origin=first["departure"]["iataCode"],
                    destination=last["arrival"]["iataCode"],
                    stops=stops
                ))

            options.append(FlightOption(
                id=f"amadeus_{i}",
                price_total=price_total,
                currency=currency,
                slices=slices,
                cabin=offer.get("travelerPricings",[{}])[0].get("fareDetailsBySegment",[{}])[0].get("cabin","ECONOMY"),
            ))
        return options

def _iso8601_duration_to_minutes(dur: str) -> int:
    # crude parser for PT#H#M
    h, m = 0, 0
    t = dur.upper().replace("PT","")
    if "H" in t:
        h_part, t = t.split("H", 1)
        h = int(h_part)
    if "M" in t:
        m_part = t.split("M", 1)[0]
        m = int(m_part)
    return h*60 + m

# ---------- Agent 3: Ranking ("find and sort best models/options") ----------
class Ranker:
    def __init__(self, price_weight=0.55, duration_weight=0.35, stops_weight=0.10):
        self.w_price = price_weight
        self.w_dur = duration_weight
        self.w_stops = stops_weight

    def rank(self, options: List[FlightOption]) -> List[Tuple[FlightOption, float]]:
        if not options:
            return []
        prices = [o.price_total for o in options]
        durs = [o.total_duration_minutes for o in options]
        stops = [o.total_stops for o in options]

        # Min-max normalize with small epsilon
        def norm(x, xs):
            mn, mx = min(xs), max(xs)
            return 0.0 if mx == mn else (x - mn) / (mx - mn)

        ranked = []
        for o in options:
            s_price = 1 - norm(o.price_total, prices)    # lower price is better
            s_dur   = 1 - norm(o.total_duration_minutes, durs)
            s_stop  = 1 - norm(o.total_stops, stops)
            score = self.w_price*s_price + self.w_dur*s_dur + self.w_stops*s_stop
            ranked.append((o, score))

        ranked.sort(key=lambda t: t[1], reverse=True)
        return ranked

# ---------- Agent 4: Rewording / presentation ----------
class Presenter:
    def __init__(self, llm: LLM):
        self.llm = llm

    def render(self, query: TravelQuery, ranked: List[Tuple[FlightOption, float]], top_k=5) -> str:
        top = ranked[:top_k]
        data = []
        for o, score in top:
            legs = []
            for s in o.slices:
                legs.append({
                    "from": s.origin,
                    "to": s.destination,
                    "depart_local": s.depart_at.isoformat(),
                    "arrive_local": s.arrive_at.isoformat(),
                    "carrier": f"{s.marketing_carrier}{s.flight_number}",
                    "stops": s.stops,
                    "duration_min": s.duration_minutes
                })
            data.append({
                "id": o.id,
                "price": f"{o.currency} {o.price_total:.2f}",
                "duration_min_total": o.total_duration_minutes,
                "stops_total": o.total_stops,
                "cabin": o.cabin,
                "legs": legs,
                "score": round(score, 3),
            })

        sys = ("You are a travel concierge. Present flight options clearly with bullets. "
               "Start with a 1–2 sentence summary, then list 3–5 best options, each with price, total duration, total stops, and key legs. "
               "Be concise and friendly. Avoid hallucinating or changing any numbers.")
        user = (
            f"Query:\n"
            f"- From: {query.origin}\n- To: {query.destination}\n"
            f"- Depart: {query.depart_date}\n"
            f"- Return: {query.return_date or 'N/A'}\n"
            f"- Pax: {query.passengers}\n"
            f"- Cabin: {query.cabin}\n\n"
            f"Top options (JSON):\n{json.dumps(data, indent=2)}"
        )
        return self.llm.rewrite(sys, user)

# ---------- Orchestrator ----------
class TravelPlanner:
    def __init__(self, flight_api: FlightAPI, llm: Optional[LLM] = None):
        self.llm = llm or LLM()
        self.nlu = NLUAgent(self.llm)
        self.flight_api = flight_api
        self.ranker = Ranker()
        self.presenter = Presenter(self.llm)

    def plan(self, user_text: str) -> dict:
        # 1) understand need
        query = self.nlu.parse(user_text)

        # 2) call flight API
        options = self.flight_api.search(query)

        # 3) sort best models/options
        ranked = self.ranker.rank(options)

        # 4) reword & present
        summary = self.presenter.render(query, ranked)

        return {
            "query": query.dict(),
            "results": [
                {
                    "id": o.id,
                    "price_total": o.price_total,
                    "currency": o.currency,
                    "duration_minutes_total": o.total_duration_minutes,
                    "stops_total": o.total_stops,
                    "cabin": o.cabin,
                    "score": score,
                    "slices": [s.dict() for s in o.slices],
                }
                for o, score in ranked
            ],
            "presentation": summary
        }

# ---------- Example usage ----------
if __name__ == "__main__":
    """
    Example:
    user_text = "I need a trip from NYC to Paris leaving Sept 20 for 6 days, 2 adults, premium economy."
    """
    user_text = input("Describe your trip: ").strip()
    planner = TravelPlanner(flight_api=AmadeusFlightAPI())
    out = planner.plan(user_text)
    print("\n=== USER PRESENTATION ===\n")
    print(out["presentation"])
    print("\n=== STRUCTURED RESULTS ===\n")
    print(json.dumps(out["results"][:3], indent=2, default=str))