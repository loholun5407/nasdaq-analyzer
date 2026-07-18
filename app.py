"""
NASDAQ Stock Analyzer - Backend (FastAPI + yfinance)
Fetches financial data, explains metrics, and rates stocks 1-10.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import yfinance as yf
import numpy as np
import os
import json
import math
from typing import Optional
from datetime import datetime, timedelta
import asyncio
import threading
import time

app = FastAPI(title="NASDAQ Stock Analyzer", version="1.0")

import pathlib
if pathlib.Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
# ─── METRIC EXPLANATIONS ───
METRIC_EXPLANATIONS = {
    "pe_ratio": {
        "name": "P/E Ratio (市盈率)",
        "formula": "股價 ÷ 每股盈利 (EPS)",
        "meaning": "投資者願意為每 $1 盈利付出幾多錢。低 P/E 可能表示股票被低估，高 P/E 表示市場預期高增長。標普 500 歷史平均約 15-20。",
        "good": "低於行業平均（通常 < 15 為平，< 10 為極平）",
        "icon": "📊"
    },
    "forward_pe": {
        "name": "Forward P/E (預測市盈率)",
        "formula": "股價 ÷ 預測未來 12 個月 EPS",
        "meaning": "用分析師預測嘅未來盈利計算嘅市盈率，比傳統 P/E 更有前瞻性。低於傳統 P/E 表示盈利預期增長。",
        "good": "低於傳統 P/E，表示盈利正在改善",
        "icon": "🔮"
    },
    "peg_ratio": {
        "name": "PEG Ratio (市盈增長率)",
        "formula": "P/E ÷ 盈利增長率 (%)",
        "meaning": "衡量估值相對於增長速度。PEG < 1 表示股票可能被低估（增長快但估值低），PEG > 2 可能過熱。",
        "good": "< 1.0 為理想，< 0.5 為極吸引",
        "icon": "🚀"
    },
    "pb_ratio": {
        "name": "P/B Ratio (市帳率)",
        "formula": "股價 ÷ 每股帳面值",
        "meaning": "市值對比公司資產淨值。低 P/B (< 1) 表示股票可能低於清算價值。銀行/金融股特別睇呢個。",
        "good": "< 1.0 可能被低估，< 3 屬合理",
        "icon": "📚"
    },
    "revenue_growth": {
        "name": "Revenue Growth (營收增長率)",
        "formula": "(今年營收 - 去年營收) ÷ 去年營收 × 100%",
        "meaning": "公司收入嘅按年增長。持續雙位數增長係好信號，負增長可能反映市場份額流失或行業衰退。",
        "good": "> 10% 強勁，> 20% 極優秀",
        "icon": "📈"
    },
    "earnings_growth": {
        "name": "Earnings Growth (盈利增長率)",
        "formula": "(今年盈利 - 去年盈利) ÷ 去年盈利 × 100%",
        "meaning": "利潤嘅按年增長。比營收增長更重要，因為反映公司真正賺錢能力。但要留意一次性收益。",
        "good": "> 10% 強勁，> 20% 極優秀",
        "icon": "💰"
    },
    "profit_margin": {
        "name": "Profit Margin (利潤率)",
        "formula": "淨利潤 ÷ 營收 × 100%",
        "meaning": "每 $100 收入有幾多係真正利潤。高利潤率 (>20%) 表示公司有定價能力同競爭優勢（護城河）。",
        "good": "> 20% 優秀，> 10% 合理",
        "icon": "💎"
    },
    "roe": {
        "name": "ROE (股東回報率)",
        "formula": "淨利潤 ÷ 股東權益 × 100%",
        "meaning": "公司用股東資金賺錢嘅效率。巴菲特最睇嘅指標之一。長期 ROE > 15% 表示優秀管理。",
        "good": "> 15% 優秀，> 20% 極佳",
        "icon": "🏆"
    },
    "debt_to_equity": {
        "name": "Debt/Equity (負債比率)",
        "formula": "總負債 ÷ 股東權益",
        "meaning": "公司用幾多借貸去營運。過高 (>2) 表示財務風險大，過低可能表示冇善用槓桿。",
        "good": "< 1.0 穩健，< 0.5 極保守",
        "icon": "⚖️"
    },
    "current_ratio": {
        "name": "Current Ratio (流動比率)",
        "formula": "流動資產 ÷ 流動負債",
        "meaning": "公司短期償債能力。> 2 表示健康，< 1 可能面臨流動性危機。",
        "good": "> 1.5 健康，> 2.0 極穩健",
        "icon": "🛡️"
    },
    "free_cash_flow": {
        "name": "Free Cash Flow (自由現金流)",
        "formula": "經營現金流 - 資本支出",
        "meaning": "公司扣除所有開支後真正可以自由使用嘅現金。正 FCF 可以回購、派息、再投資。負 FCF 係紅色警報。",
        "good": "正值且持續增長",
        "icon": "💵"
    },
    "beta": {
        "name": "Beta (波動系數)",
        "formula": "股票相對於大市嘅波動性",
        "meaning": "Beta = 1 表示跟大市同步。> 1 更大波幅（科技股常見），< 1 較防守（公用股）。",
        "good": "視乎風險偏好：保守揀 < 1，進取可接受 > 1.5",
        "icon": "🎢"
    },
    "dividend_yield": {
        "name": "Dividend Yield (股息率)",
        "formula": "每年股息 ÷ 股價 × 100%",
        "meaning": "每年收息回報。高息 (>4%) 可能吸引但小心係「股息陷阱」（股價大跌所致）。",
        "good": "> 2% 合理，> 4% 高息但要查 payout ratio",
        "icon": "💸"
    },
    "market_cap": {
        "name": "Market Cap (市值)",
        "formula": "股價 × 流通股數",
        "meaning": "公司總市值。Mega Cap (>$200B), Large Cap ($10B-$200B), Mid Cap ($2B-$10B), Small Cap (<$2B)。",
        "good": "大市值通常較穩定",
        "icon": "🏗️"
    }
}


# ─── HELPER FUNCTIONS ───

def safe_float(val, default=None):
    """Safely convert to float, return default if None/NaN."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 2)
    except (ValueError, TypeError):
        return default


def safe_int(val, default=None):
    """Safely convert to int."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def format_large_number(num):
    """Format large numbers to human readable."""
    if num is None:
        return "N/A"
    num = float(num)
    if abs(num) >= 1e12:
        return f"${num/1e12:.2f}T"
    elif abs(num) >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"${num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"


def fetch_stock_data(ticker: str):
    """Fetch comprehensive stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or info.get("regularMarketPrice") is None and info.get("regularMarketPrice") is None:
            # For crypto, check different price field
            price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("regularMarketOpen")
            if price is None:
                return None

        data = {}
        data["ticker"] = ticker.upper()
        data["name"] = info.get("longName") or info.get("shortName") or info.get("name") or ticker
        data["sector"] = info.get("sector", info.get("category", "N/A"))
        data["industry"] = info.get("industry", info.get("category", "N/A"))
        data["country"] = info.get("country", "N/A")
        data["website"] = info.get("website", "")
        data["description"] = (info.get("longBusinessSummary") or info.get("description") or "N/A")[:500]

        # Price data - handle both stocks and crypto
        data["price"] = safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
        data["prev_close"] = safe_float(info.get("regularMarketPreviousClose") or info.get("previousClose"))
        data["change"] = safe_float(data["price"] - data["prev_close"]) if data["price"] and data["prev_close"] else None
        data["change_pct"] = safe_float((data["change"] / data["prev_close"] * 100)) if data["change"] and data["prev_close"] else None

        data["52w_high"] = safe_float(info.get("fiftyTwoWeekHigh"))
        data["52w_low"] = safe_float(info.get("fiftyTwoWeekLow"))
        data["50d_ma"] = safe_float(info.get("fiftyDayAverage"))
        data["200d_ma"] = safe_float(info.get("twoHundredDayAverage"))
        data["volume"] = safe_int(info.get("regularMarketVolume") or info.get("volume24Hr"))
        data["avg_volume"] = safe_int(info.get("averageVolume") or info.get("averageVolume10days"))

        # Market Cap
        data["market_cap"] = safe_float(info.get("marketCap"))
        data["market_cap_fmt"] = format_large_number(data["market_cap"]) if data["market_cap"] else ("$" + format_large_number(info.get("totalAssets", 0)) if info.get("totalAssets") else "N/A")
        data["enterprise_value"] = safe_float(info.get("enterpriseValue"))
        data["shares_outstanding"] = safe_float(info.get("sharesOutstanding"))

        # Valuation
        data["pe_ratio"] = safe_float(info.get("trailingPE"))
        data["forward_pe"] = safe_float(info.get("forwardPE"))
        data["peg_ratio"] = safe_float(info.get("pegRatio"))
        data["pb_ratio"] = safe_float(info.get("priceToBook"))
        data["ps_ratio"] = safe_float(info.get("priceToSalesTrailing12Months"))
        data["ev_to_ebitda"] = safe_float(info.get("enterpriseToEbitda"))
        data["ev_to_revenue"] = safe_float(info.get("enterpriseToRevenue"))

        # Growth
        data["revenue_growth"] = safe_float(info.get("revenueGrowth"))
        if data["revenue_growth"] is not None:
            data["revenue_growth"] = round(data["revenue_growth"] * 100, 2)
        data["earnings_growth"] = safe_float(info.get("earningsGrowth"))
        if data["earnings_growth"] is not None:
            data["earnings_growth"] = round(data["earnings_growth"] * 100, 2)
        data["earnings_quarterly_growth"] = safe_float(info.get("earningsQuarterlyGrowth"))
        if data["earnings_quarterly_growth"] is not None:
            data["earnings_quarterly_growth"] = round(data["earnings_quarterly_growth"] * 100, 2)

        # Profitability
        data["profit_margin"] = safe_float(info.get("profitMargins"))
        if data["profit_margin"] is not None:
            data["profit_margin"] = round(data["profit_margin"] * 100, 2)
        data["operating_margin"] = safe_float(info.get("operatingMargins"))
        if data["operating_margin"] is not None:
            data["operating_margin"] = round(data["operating_margin"] * 100, 2)
        data["gross_margin"] = safe_float(info.get("grossMargins"))
        if data["gross_margin"] is not None:
            data["gross_margin"] = round(data["gross_margin"] * 100, 2)

        data["roe"] = safe_float(info.get("returnOnEquity"))
        if data["roe"] is not None:
            data["roe"] = round(data["roe"] * 100, 2)
        data["roa"] = safe_float(info.get("returnOnAssets"))
        if data["roa"] is not None:
            data["roa"] = round(data["roa"] * 100, 2)

        # Financial Health
        data["debt_to_equity"] = safe_float(info.get("debtToEquity"))
        data["current_ratio"] = safe_float(info.get("currentRatio"))
        data["quick_ratio"] = safe_float(info.get("quickRatio"))
        data["total_cash"] = safe_float(info.get("totalCash"))
        data["total_debt"] = safe_float(info.get("totalDebt"))
        data["net_cash"] = safe_float(data["total_cash"] - data["total_debt"]) if data["total_cash"] is not None and data["total_debt"] is not None else None

        # Cash Flow
        data["free_cash_flow"] = safe_float(info.get("freeCashflow"))
        data["operating_cash_flow"] = safe_float(info.get("operatingCashflow"))
        data["fcf_yield"] = None
        if data["free_cash_flow"] and data["market_cap"] and data["market_cap"] > 0:
            data["fcf_yield"] = round(data["free_cash_flow"] / data["market_cap"] * 100, 2)

        # Dividends
        data["dividend_yield"] = safe_float(info.get("dividendYield"))
        if data["dividend_yield"] is not None:
            data["dividend_yield"] = round(data["dividend_yield"] * 100, 2)
        data["dividend_rate"] = safe_float(info.get("dividendRate"))
        data["payout_ratio"] = safe_float(info.get("payoutRatio"))

        # Risk
        data["beta"] = safe_float(info.get("beta"))
        data["short_ratio"] = safe_float(info.get("shortRatio"))
        data["short_pct"] = safe_float(info.get("shortPercentOfFloat"))
        if data["short_pct"] is not None:
            data["short_pct"] = round(data["short_pct"] * 100, 2)

        # EPS
        data["eps_ttm"] = safe_float(info.get("trailingEps"))
        data["eps_forward"] = safe_float(info.get("forwardEps"))

        # Revenue & Earnings
        data["total_revenue"] = safe_float(info.get("totalRevenue"))
        data["total_revenue_fmt"] = format_large_number(data["total_revenue"]) if data["total_revenue"] else "N/A"
        data["net_income"] = safe_float(info.get("netIncomeToCommon"))
        data["net_income_fmt"] = format_large_number(data["net_income"]) if data["net_income"] else "N/A"
        data["ebitda"] = safe_float(info.get("ebitda"))
        data["ebitda_fmt"] = format_large_number(data["ebitda"]) if data["ebitda"] else "N/A"

        # Analyst
        data["target_mean"] = safe_float(info.get("targetMeanPrice"))
        data["target_high"] = safe_float(info.get("targetHighPrice"))
        data["target_low"] = safe_float(info.get("targetLowPrice"))
        data["recommendation"] = info.get("recommendationKey", "N/A")
        data["num_analysts"] = safe_int(info.get("numberOfAnalystOpinions"))

        # Next earnings
        data["next_earnings_date"] = None
        try:
            calendar = stock.calendar
            if calendar is not None and not calendar.empty:
                for col in calendar.columns:
                    if 'earnings' in str(col).lower():
                        dt = calendar.iloc[0][col]
                        data["next_earnings_date"] = str(dt)[:10] if dt else None
                        break
        except:
            pass

        data["upside"] = None
        if data["target_mean"] and data["price"] and data["price"] > 0:
            data["upside"] = round((data["target_mean"] - data["price"]) / data["price"] * 100, 2)

        # News
        data["news"] = []
        try:
            stock_news = stock.news
            if stock_news:
                for n in stock_news[:10]:
                    c = n.get("content", n) if isinstance(n, dict) else {}
                    news_item = {
                        "title": (c.get("title") or "N/A")[:150],
                        "publisher": c.get("provider", {}).get("displayName", "Yahoo Finance") if isinstance(c.get("provider"), dict) else "Yahoo Finance",
                        "link": c.get("canonicalUrl", {}).get("url", c.get("clickThroughUrl", {}).get("url", "")) if isinstance(c.get("canonicalUrl"), dict) else "",
                        "published_str": c.get("pubDate", c.get("displayTime", ""))[:16] if c.get("pubDate") else "N/A",
                        "summary": (c.get("summary") or "")[:200],
                    }
                    if news_item["title"] != "N/A":
                        data["news"].append(news_item)
        except:
            pass

        # Analyst Recommendations
        data["recommendations_trend"] = []
        try:
            recs = stock.recommendations
            if recs is not None and not recs.empty and "period" in recs.columns:
                latest = recs.iloc[-1]
                data["recommendations_trend"] = {
                    "period": str(latest.get("period", "N/A")),
                    "strongBuy": safe_int(latest.get("strongBuy", 0)),
                    "buy": safe_int(latest.get("buy", 0)),
                    "hold": safe_int(latest.get("hold", 0)),
                    "sell": safe_int(latest.get("sell", 0)),
                    "strongSell": safe_int(latest.get("strongSell", 0)),
                    "total": 0,
                }
                data["recommendations_trend"]["total"] = sum([
                    data["recommendations_trend"]["strongBuy"],
                    data["recommendations_trend"]["buy"],
                    data["recommendations_trend"]["hold"],
                    data["recommendations_trend"]["sell"],
                    data["recommendations_trend"]["strongSell"]
                ])
        except:
            pass

        # Insider Transactions
        data["insider_transactions"] = []
        try:
            insider = stock.insider_transactions
            if insider is not None and not insider.empty:
                for _, row in insider.head(10).iterrows():
                    tx = {
                        "insider": str(row.get("Insider", row.get(" insider", "N/A"))),
                        "title": str(row.get("Title", row.get(" title", "N/A"))),
                        "transaction": str(row.get("Transaction", row.get(" transaction", "N/A"))),
                        "shares": safe_int(row.get("Shares", row.get(" shares", 0))),
                        "value": safe_float(row.get("Value", row.get(" value", 0))),
                        "date": str(row.get("Date", row.get(" date", "N/A"))),
                    }
                    if tx["shares"] and tx["shares"] > 0:
                        data["insider_transactions"].append(tx)
        except:
            pass

        # Peers
        data["peers"] = []
        try:
            peer_groups = {
                "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMD", "INTC", "ADBE"],
                "Financial": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "V", "MA"],
                "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "ABT", "AMGN"],
                "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "BKNG", "MAR"],
                "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST", "PM", "WBA"],
                "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
                "Communication": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ"],
                "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD", "XRP-USD"],
            }
            sector = info.get("sector", info.get("category", ""))
            peer_tickers = peer_groups.get(sector, ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA"])
            peer_tickers = [t for t in peer_tickers if t != ticker.upper()][:8]
            
            for pt in peer_tickers[:8]:
                try:
                    pstock = yf.Ticker(pt)
                    pinfo = pstock.info
                    if pinfo and (pinfo.get("regularMarketPrice") or pinfo.get("currentPrice")):
                        peer_data = {
                            "ticker": pt,
                            "name": pinfo.get("shortName", pt),
                            "price": safe_float(pinfo.get("regularMarketPrice") or pinfo.get("currentPrice")),
                            "market_cap_fmt": format_large_number(safe_float(pinfo.get("marketCap"))) if pinfo.get("marketCap") else "N/A",
                            "pe_ratio": safe_float(pinfo.get("trailingPE")),
                            "revenue_growth": safe_float(pinfo.get("revenueGrowth")),
                            "profit_margin": safe_float(pinfo.get("profitMargins")),
                        }
                        if peer_data["revenue_growth"] is not None:
                            peer_data["revenue_growth"] = round(peer_data["revenue_growth"] * 100, 2)
                        if peer_data["profit_margin"] is not None:
                            peer_data["profit_margin"] = round(peer_data["profit_margin"] * 100, 2)
                        data["peers"].append(peer_data)
                except:
                    pass
        except:
            pass

        # Earnings Dates
        data["earnings_dates"] = []
        try:
            earnings = stock.earnings_dates
            if earnings is not None and not earnings.empty:
                for idx, row in earnings.head(4).iterrows():
                    ed = {
                        "date": str(idx)[:10] if hasattr(idx, 'strftime') else str(idx),
                        "eps_estimate": safe_float(row.get("EPS Estimate", row.get(" eps estimate", None))),
                        "eps_actual": safe_float(row.get("EPS Actual", row.get(" eps actual", None))),
                        "surprise": safe_float(row.get("Surprise(%)", row.get(" surprise(%)", None))),
                    }
                    data["earnings_dates"].append(ed)
        except:
            pass

        # Rating
        rating_result = calculate_rating(data)
        data["rating"] = rating_result["rating"]
        data["rating_breakdown"] = get_rating_breakdown(data, rating_result)

        return data

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def calculate_rating(data):
    """Calculate 1-10 rating: Valuation(30%) + Growth(25%) + Profitability(25%) + Health(20%)."""
    scores = []

    # VALUATION (30%)
    val_scores = []
    pe = data.get("pe_ratio")
    if pe is not None:
        if pe < 0: val_scores.append(0.1)
        elif pe <= 10: val_scores.append(0.8)
        elif pe <= 20: val_scores.append(1.0)
        elif pe <= 30: val_scores.append(0.7)
        elif pe <= 50: val_scores.append(0.4)
        else: val_scores.append(0.1)
    else: val_scores.append(0.5)

    peg = data.get("peg_ratio")
    if peg is not None:
        if peg < 0: val_scores.append(0.3)
        elif peg <= 0.5: val_scores.append(1.0)
        elif peg <= 1.0: val_scores.append(0.9)
        elif peg <= 1.5: val_scores.append(0.7)
        elif peg <= 2.0: val_scores.append(0.5)
        else: val_scores.append(0.2)
    else: val_scores.append(0.5)

    pb = data.get("pb_ratio")
    if pb is not None:
        if pb < 0: val_scores.append(0.2)
        elif pb <= 1: val_scores.append(0.9)
        elif pb <= 3: val_scores.append(0.8)
        elif pb <= 5: val_scores.append(0.6)
        else: val_scores.append(0.3)
    else: val_scores.append(0.5)

    val_avg = sum(val_scores) / len(val_scores) if val_scores else 0.5
    scores.append(("Valuation", val_avg, 0.30))

    # GROWTH (25%)
    growth_scores = []
    rev_g = data.get("revenue_growth")
    if rev_g is not None:
        if rev_g > 30: growth_scores.append(1.0)
        elif rev_g > 20: growth_scores.append(0.9)
        elif rev_g > 10: growth_scores.append(0.7)
        elif rev_g > 5: growth_scores.append(0.6)
        elif rev_g > 0: growth_scores.append(0.4)
        else: growth_scores.append(0.1)
    else: growth_scores.append(0.5)

    earn_g = data.get("earnings_growth")
    if earn_g is not None:
        if earn_g > 30: growth_scores.append(1.0)
        elif earn_g > 20: growth_scores.append(0.9)
        elif earn_g > 10: growth_scores.append(0.7)
        elif earn_g > 5: growth_scores.append(0.6)
        elif earn_g > 0: growth_scores.append(0.4)
        else: growth_scores.append(0.1)
    else: growth_scores.append(0.5)

    growth_avg = sum(growth_scores) / len(growth_scores) if growth_scores else 0.5
    scores.append(("Growth", growth_avg, 0.25))

    # PROFITABILITY (25%)
    prof_scores = []
    pm = data.get("profit_margin")
    if pm is not None:
        if pm > 25: prof_scores.append(1.0)
        elif pm > 20: prof_scores.append(0.9)
        elif pm > 15: prof_scores.append(0.8)
        elif pm > 10: prof_scores.append(0.7)
        elif pm > 5: prof_scores.append(0.5)
        elif pm > 0: prof_scores.append(0.3)
        else: prof_scores.append(0.05)
    else: prof_scores.append(0.5)

    roe = data.get("roe")
    if roe is not None:
        if roe > 30: prof_scores.append(1.0)
        elif roe > 20: prof_scores.append(0.9)
        elif roe > 15: prof_scores.append(0.8)
        elif roe > 10: prof_scores.append(0.6)
        elif roe > 5: prof_scores.append(0.4)
        elif roe > 0: prof_scores.append(0.2)
        else: prof_scores.append(0.05)
    else: prof_scores.append(0.5)

    prof_avg = sum(prof_scores) / len(prof_scores) if prof_scores else 0.5
    scores.append(("Profitability", prof_avg, 0.25))

    # FINANCIAL HEALTH (20%)
    health_scores = []
    de = data.get("debt_to_equity")
    if de is not None:
        if de < 0: health_scores.append(0.3)
        elif de <= 20: health_scores.append(1.0)
        elif de <= 50: health_scores.append(0.9)
        elif de <= 80: health_scores.append(0.7)
        elif de <= 120: health_scores.append(0.5)
        else: health_scores.append(0.2)
    else: health_scores.append(0.5)

    fcf_y = data.get("fcf_yield")
    if fcf_y is not None:
        if fcf_y > 10: health_scores.append(1.0)
        elif fcf_y > 5: health_scores.append(0.9)
        elif fcf_y > 3: health_scores.append(0.7)
        elif fcf_y > 1: health_scores.append(0.5)
        elif fcf_y > 0: health_scores.append(0.3)
        else: health_scores.append(0.1)
    else: health_scores.append(0.5)

    health_avg = sum(health_scores) / len(health_scores) if health_scores else 0.5
    scores.append(("Financial Health", health_avg, 0.20))

    categories = {}
    for name, cat_score, weight in scores:
        key = name.lower().replace(' ', '_')
        categories[key] = round(cat_score, 3)

    total = sum(score * weight for _, score, weight in scores)
    rating = round(total * 10, 1)

    return {
        "rating": max(1.0, min(10.0, rating)),
        "valuation": categories.get("valuation", 0.5),
        "growth": categories.get("growth", 0.5),
        "profitability": categories.get("profitability", 0.5),
        "financial_health": categories.get("financial_health", 0.5),
    }


def get_rating_breakdown(data, rating_result=None):
    """Get detailed rating breakdown with reasoning."""
    pe = data.get("pe_ratio")
    peg = data.get("peg_ratio")
    rev_g = data.get("revenue_growth")
    earn_g = data.get("earnings_growth")
    pm = data.get("profit_margin")
    roe = data.get("roe")
    de = data.get("debt_to_equity")
    fcf_y = data.get("fcf_yield")

    positives, negatives, neutrals = [], [], []

    if pe is not None:
        if pe < 0: negatives.append(f"P/E 為負數 ({pe})，錄得虧損 ⚠️")
        elif pe < 15: positives.append(f"P/E 偏低 ({pe})，估值吸引 👍")
        elif pe < 25: neutrals.append(f"P/E 合理 ({pe})")
        else: negatives.append(f"P/E 偏高 ({pe})，估值較貴 ⚠️")

    if peg is not None:
        if 0 < peg < 1: positives.append(f"PEG = {peg} (<1)，增長速度超過估值 🚀")
        elif 1 <= peg < 2: positives.append(f"PEG = {peg}，估值合理配合增長 ✅")
        elif peg >= 2: neutrals.append(f"PEG = {peg} (>2)")

    if rev_g is not None:
        if rev_g > 20: positives.append(f"營收增長強勁 ({rev_g}%) 📈")
        elif rev_g > 5: neutrals.append(f"營收溫和增長 ({rev_g}%)")
        elif rev_g <= 0: negatives.append(f"營收收縮 ({rev_g}%) 📉")

    if earn_g is not None:
        if earn_g > 20: positives.append(f"盈利增長強勁 ({earn_g}%) 💰")
        elif earn_g > 5: neutrals.append(f"盈利溫和增長 ({earn_g}%)")
        elif earn_g <= 0: negatives.append(f"盈利下滑 ({earn_g}%) 📉")

    if pm is not None:
        if pm > 20: positives.append(f"利潤率優秀 ({pm}%) 💎")
        elif pm > 10: neutrals.append(f"利潤率合理 ({pm}%)")
        elif pm <= 0: negatives.append(f"利潤率為負 ({pm}%) 🔴")

    if roe is not None:
        if roe > 20: positives.append(f"ROE 極佳 ({roe}%) 🏆")
        elif roe > 10: neutrals.append(f"ROE 合理 ({roe}%)")
        elif roe <= 0: negatives.append(f"ROE 為負 ({roe}%)")

    if de is not None:
        if de < 30: positives.append(f"負債比率極低 ({de}%) 🛡️")
        elif de < 80: neutrals.append(f"負債比率適中 ({de}%)")
        elif de > 120: negatives.append(f"負債比率偏高 ({de}%) ⚖️")

    if fcf_y is not None:
        if fcf_y > 5: positives.append(f"FCF Yield 強勁 ({fcf_y}%) 💵")
        elif fcf_y > 0: neutrals.append(f"FCF Yield 正面 ({fcf_y}%)")
        elif fcf_y <= 0: negatives.append(f"FCF Yield 為負 ({fcf_y}%) 🔴")

    result = {"positives": positives, "negatives": negatives, "neutrals": neutrals}
    if rating_result:
        result["valuation"] = rating_result.get("valuation", 0.5)
        result["growth"] = rating_result.get("growth", 0.5)
        result["profitability"] = rating_result.get("profitability", 0.5)
        result["financial_health"] = rating_result.get("financial_health", 0.5)
    return result



def get_all_tickers():
    """Return curated lists of stocks across markets: NASDAQ, S&P500, DOW, Crypto."""
    nasdaq = [
        ("AAPL", "Apple Inc.", "Technology", "NASDAQ"),
        ("MSFT", "Microsoft Corp.", "Technology", "NASDAQ"),
        ("GOOGL", "Alphabet Inc.", "Technology", "NASDAQ"),
        ("AMZN", "Amazon.com Inc.", "Consumer Cyclical", "NASDAQ"),
        ("NVDA", "NVIDIA Corp.", "Technology", "NASDAQ"),
        ("META", "Meta Platforms Inc.", "Technology", "NASDAQ"),
        ("TSLA", "Tesla Inc.", "Consumer Cyclical", "NASDAQ"),
        ("AVGO", "Broadcom Inc.", "Technology", "NASDAQ"),
        ("COST", "Costco Wholesale", "Consumer Defensive", "NASDAQ"),
        ("NFLX", "Netflix Inc.", "Communication", "NASDAQ"),
        ("AMD", "AMD Inc.", "Technology", "NASDAQ"),
        ("ADBE", "Adobe Inc.", "Technology", "NASDAQ"),
        ("PEP", "PepsiCo Inc.", "Consumer Defensive", "NASDAQ"),
        ("INTC", "Intel Corp.", "Technology", "NASDAQ"),
        ("CSCO", "Cisco Systems", "Technology", "NASDAQ"),
        ("CMCSA", "Comcast Corp.", "Communication", "NASDAQ"),
        ("QCOM", "Qualcomm Inc.", "Technology", "NASDAQ"),
        ("TXN", "Texas Instruments", "Technology", "NASDAQ"),
        ("AMGN", "Amgen Inc.", "Healthcare", "NASDAQ"),
        ("INTU", "Intuit Inc.", "Technology", "NASDAQ"),
        ("ISRG", "Intuitive Surgical", "Healthcare", "NASDAQ"),
        ("BKNG", "Booking Holdings", "Consumer Cyclical", "NASDAQ"),
        ("GILD", "Gilead Sciences", "Healthcare", "NASDAQ"),
        ("SBUX", "Starbucks Corp.", "Consumer Cyclical", "NASDAQ"),
        ("ADI", "Analog Devices", "Technology", "NASDAQ"),
        ("LRCX", "Lam Research", "Technology", "NASDAQ"),
        ("MU", "Micron Technology", "Technology", "NASDAQ"),
        ("REGN", "Regeneron Pharma", "Healthcare", "NASDAQ"),
        ("VRTX", "Vertex Pharma", "Healthcare", "NASDAQ"),
        ("KLAC", "KLA Corp.", "Technology", "NASDAQ"),
        ("SNPS", "Synopsys Inc.", "Technology", "NASDAQ"),
        ("CDNS", "Cadence Design", "Technology", "NASDAQ"),
        ("ASML", "ASML Holding", "Technology", "NASDAQ"),
        ("PYPL", "PayPal Holdings", "Financial", "NASDAQ"),
        ("ABNB", "Airbnb Inc.", "Consumer Cyclical", "NASDAQ"),
        ("MRVL", "Marvell Technology", "Technology", "NASDAQ"),
        ("WDAY", "Workday Inc.", "Technology", "NASDAQ"),
        ("CRWD", "CrowdStrike Holdings", "Technology", "NASDAQ"),
        ("DASH", "DoorDash Inc.", "Technology", "NASDAQ"),
        ("TTD", "Trade Desk", "Technology", "NASDAQ"),
        ("DDOG", "Datadog Inc.", "Technology", "NASDAQ"),
        ("ZS", "Zscaler Inc.", "Technology", "NASDAQ"),
        ("MDB", "MongoDB Inc.", "Technology", "NASDAQ"),
        ("PLTR", "Palantir Tech", "Technology", "NASDAQ"),
        ("FTNT", "Fortinet Inc.", "Technology", "NASDAQ"),
        ("PANW", "Palo Alto Networks", "Technology", "NASDAQ"),
        ("ADSK", "Autodesk Inc.", "Technology", "NASDAQ"),
        ("ORLY", "O'Reilly Auto", "Consumer Cyclical", "NASDAQ"),
        ("MAR", "Marriott Intl", "Consumer Cyclical", "NASDAQ"),
        ("DKNG", "DraftKings Inc.", "Consumer Cyclical", "NASDAQ"),
        ("RIVN", "Rivian Automotive", "Consumer Cyclical", "NASDAQ"),
        ("SOFI", "SoFi Technologies", "Financial", "NASDAQ"),
        ("HOOD", "Robinhood Markets", "Financial", "NASDAQ"),
        ("RBLX", "Roblox Corp.", "Technology", "NASDAQ"),
        ("SNAP", "Snap Inc.", "Communication", "NASDAQ"),
        ("ZM", "Zoom Video", "Technology", "NASDAQ"),
        ("DOCU", "DocuSign Inc.", "Technology", "NASDAQ"),
        ("OKTA", "Okta Inc.", "Technology", "NASDAQ"),
        ("TWLO", "Twilio Inc.", "Technology", "NASDAQ"),
        ("SPOT", "Spotify Technology", "Communication", "NASDAQ"),
        ("COIN", "Coinbase Global", "Financial", "NASDAQ"),
        ("MELI", "MercadoLibre", "Consumer Cyclical", "NASDAQ"),
        ("TEAM", "Atlassian Corp.", "Technology", "NASDAQ"),
        ("SNOW", "Snowflake Inc.", "Technology", "NASDAQ"),
        ("UBER", "Uber Technologies", "Technology", "NASDAQ"),
        # --- 以下為 2026-07-17 新增 ---
        ("SPCX", "SpaceX", "Industrials", "NASDAQ"),
        ("TSM", "TSMC Taiwan", "Technology", "NASDAQ"),
        ("IONQ", "IonQ Inc.", "Technology", "NASDAQ"),
        ("RGTI", "Rigetti Computing", "Technology", "NASDAQ"),
        ("QBTS", "D-Wave Quantum", "Technology", "NASDAQ"),
        ("QUBT", "Quantum Computing Inc", "Technology", "NASDAQ"),
        ("RDDT", "Reddit Inc.", "Communication", "NASDAQ"),
        ("ARM", "Arm Holdings", "Technology", "NASDAQ"),
        ("ASTS", "AST SpaceMobile", "Technology", "NASDAQ"),
        ("RKLB", "Rocket Lab", "Industrials", "NASDAQ"),
        ("SMCI", "Super Micro Computer", "Technology", "NASDAQ"),
        ("HII", "Huntington Ingalls", "Industrials", "NASDAQ"),
        ("LDOS", "Leidos Holdings", "Industrials", "NASDAQ"),
        ("CART", "Instacart (Maplebear)", "Technology", "NASDAQ"),
        ("XYZ", "XYZ Ventures", "Technology", "NASDAQ"),
        ("MSTR", "MicroStrategy", "Technology", "NASDAQ"),
        ("MARA", "MARA Holdings", "Financial", "NASDAQ"),
        ("TMDX", "TransMedics Group", "Healthcare", "NASDAQ"),
    ]
    
    sp500 = [
        ("BRK-B", "Berkshire Hathaway", "Financial", "S&P 500"),
        ("JPM", "JPMorgan Chase", "Financial", "S&P 500"),
        ("V", "Visa Inc.", "Financial", "S&P 500"),
        ("MA", "Mastercard Inc.", "Financial", "S&P 500"),
        ("JNJ", "Johnson & Johnson", "Healthcare", "S&P 500"),
        ("WMT", "Walmart Inc.", "Consumer Defensive", "S&P 500"),
        ("PG", "Procter & Gamble", "Consumer Defensive", "S&P 500"),
        ("XOM", "Exxon Mobil", "Energy", "S&P 500"),
        ("UNH", "UnitedHealth Group", "Healthcare", "S&P 500"),
        ("HD", "Home Depot", "Consumer Cyclical", "S&P 500"),
        ("BAC", "Bank of America", "Financial", "S&P 500"),
        ("KO", "Coca-Cola", "Consumer Defensive", "S&P 500"),
        ("CVX", "Chevron Corp.", "Energy", "S&P 500"),
        ("PFE", "Pfizer Inc.", "Healthcare", "S&P 500"),
        ("ABBV", "AbbVie Inc.", "Healthcare", "S&P 500"),
        ("MRK", "Merck & Co.", "Healthcare", "S&P 500"),
        ("TMO", "Thermo Fisher", "Healthcare", "S&P 500"),
        ("NKE", "Nike Inc.", "Consumer Cyclical", "S&P 500"),
        ("DIS", "Walt Disney", "Communication", "S&P 500"),
        ("ABT", "Abbott Labs", "Healthcare", "S&P 500"),
        ("DHR", "Danaher Corp.", "Healthcare", "S&P 500"),
        ("CRM", "Salesforce Inc.", "Technology", "S&P 500"),
        ("ORCL", "Oracle Corp.", "Technology", "S&P 500"),
        ("IBM", "IBM Corp.", "Technology", "S&P 500"),
        ("ACN", "Accenture PLC", "Technology", "S&P 500"),
        ("MCD", "McDonald's", "Consumer Cyclical", "S&P 500"),
        ("LIN", "Linde PLC", "Basic Materials", "S&P 500"),
        ("LLY", "Eli Lilly", "Healthcare", "S&P 500"),
        ("PM", "Philip Morris", "Consumer Defensive", "S&P 500"),
        ("WFC", "Wells Fargo", "Financial", "S&P 500"),
        ("CAT", "Caterpillar Inc.", "Industrials", "S&P 500"),
        ("GE", "GE Aerospace", "Industrials", "S&P 500"),
        ("BA", "Boeing Co.", "Industrials", "S&P 500"),
        ("RTX", "RTX Corp.", "Industrials", "S&P 500"),
        ("LMT", "Lockheed Martin", "Industrials", "S&P 500"),
        ("SPGI", "S&P Global Inc.", "Financial", "S&P 500"),
        ("BLK", "BlackRock Inc.", "Financial", "S&P 500"),
        ("GS", "Goldman Sachs", "Financial", "S&P 500"),
        ("MS", "Morgan Stanley", "Financial", "S&P 500"),
        ("AXP", "American Express", "Financial", "S&P 500"),
        ("NEE", "NextEra Energy", "Utilities", "S&P 500"),
        ("SO", "Southern Co.", "Utilities", "S&P 500"),
        ("DUK", "Duke Energy", "Utilities", "S&P 500"),
        ("UPS", "UPS Inc.", "Industrials", "S&P 500"),
        ("FDX", "FedEx Corp.", "Industrials", "S&P 500"),
        ("LOW", "Lowe's Companies", "Consumer Cyclical", "S&P 500"),
        ("TGT", "Target Corp.", "Consumer Cyclical", "S&P 500"),
        ("AMD", "AMD Inc.", "Technology", "S&P 500"),
        ("NOW", "ServiceNow Inc.", "Technology", "S&P 500"),
        ("UBER", "Uber Technologies", "Technology", "S&P 500"),
    ]
    
    dow = [
        ("AAPL", "Apple Inc.", "Technology", "DOW"),
        ("MSFT", "Microsoft Corp.", "Technology", "DOW"),
        ("JPM", "JPMorgan Chase", "Financial", "DOW"),
        ("JNJ", "Johnson & Johnson", "Healthcare", "DOW"),
        ("WMT", "Walmart Inc.", "Consumer Defensive", "DOW"),
        ("PG", "Procter & Gamble", "Consumer Defensive", "DOW"),
        ("HD", "Home Depot", "Consumer Cyclical", "DOW"),
        ("KO", "Coca-Cola", "Consumer Defensive", "DOW"),
        ("CVX", "Chevron Corp.", "Energy", "DOW"),
        ("MRK", "Merck & Co.", "Healthcare", "DOW"),
        ("NKE", "Nike Inc.", "Consumer Cyclical", "DOW"),
        ("DIS", "Walt Disney", "Communication", "DOW"),
        ("MCD", "McDonald's", "Consumer Cyclical", "DOW"),
        ("CRM", "Salesforce Inc.", "Technology", "DOW"),
        ("IBM", "IBM Corp.", "Technology", "DOW"),
        ("CAT", "Caterpillar Inc.", "Industrials", "DOW"),
        ("BA", "Boeing Co.", "Industrials", "DOW"),
        ("GS", "Goldman Sachs", "Financial", "DOW"),
        ("AXP", "American Express", "Financial", "DOW"),
        ("AMGN", "Amgen Inc.", "Healthcare", "DOW"),
        ("VZ", "Verizon Comm.", "Communication", "DOW"),
        ("HON", "Honeywell Intl", "Industrials", "DOW"),
        ("UNH", "UnitedHealth", "Healthcare", "DOW"),
        ("INTC", "Intel Corp.", "Technology", "DOW"),
        ("CSCO", "Cisco Systems", "Technology", "DOW"),
        ("TRV", "Travelers Co.", "Financial", "DOW"),
        ("MMM", "3M Company", "Industrials", "DOW"),
        ("V", "Visa Inc.", "Financial", "DOW"),
        ("DOW", "Dow Inc.", "Basic Materials", "DOW"),
        ("WBA", "Walgreens Boots", "Healthcare", "DOW"),
    ]
    
    crypto = [
        ("BTC-USD", "Bitcoin", "Crypto", "CRYPTO"),
        ("ETH-USD", "Ethereum", "Crypto", "CRYPTO"),
        ("SOL-USD", "Solana", "Crypto", "CRYPTO"),
        ("DOGE-USD", "Dogecoin", "Crypto", "CRYPTO"),
        ("ADA-USD", "Cardano", "Crypto", "CRYPTO"),
        ("XRP-USD", "Ripple XRP", "Crypto", "CRYPTO"),
        ("AVAX-USD", "Avalanche", "Crypto", "CRYPTO"),
        ("DOT-USD", "Polkadot", "Crypto", "CRYPTO"),
        ("LINK-USD", "Chainlink", "Crypto", "CRYPTO"),
        ("UNI-USD", "Uniswap", "Crypto", "CRYPTO"),
        ("MATIC-USD", "Polygon", "Crypto", "CRYPTO"),
        ("SHIB-USD", "Shiba Inu", "Crypto", "CRYPTO"),
        ("LTC-USD", "Litecoin", "Crypto", "CRYPTO"),
        ("BCH-USD", "Bitcoin Cash", "Crypto", "CRYPTO"),
        ("ATOM-USD", "Cosmos", "Crypto", "CRYPTO"),
        ("FIL-USD", "Filecoin", "Crypto", "CRYPTO"),
        ("APT-USD", "Aptos", "Crypto", "CRYPTO"),
        ("ARB-USD", "Arbitrum", "Crypto", "CRYPTO"),
        ("OP-USD", "Optimism", "Crypto", "CRYPTO"),
        ("SUI-USD", "Sui", "Crypto", "CRYPTO"),
    ]
    
    return {
        "NASDAQ": nasdaq,
        "SP500": sp500,
        "DOW": dow,
        "CRYPTO": crypto,
    }


# Legacy function for backward compatibility
def get_nasdaq_tickers():
    """Legacy: return NASDAQ tickers only."""
    return get_all_tickers()["NASDAQ"]



# ─── TECHNICAL ANALYSIS ───

def calc_rsi(prices, period=14):
    """Calculate RSI for a price series."""
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi), 1)


def calc_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD, signal line, and histogram."""
    if len(prices) < slow + signal:
        return None, None, None
    ema_fast = np.array(prices)
    ema_slow = np.array(prices)
    # Simple EMA calculation
    alpha_fast = 2 / (fast + 1)
    alpha_slow = 2 / (slow + 1)
    for i in range(1, len(ema_fast)):
        ema_fast[i] = ema_fast[i] * alpha_fast + ema_fast[i-1] * (1 - alpha_fast)
    for i in range(1, len(ema_slow)):
        ema_slow[i] = ema_slow[i] * alpha_slow + ema_slow[i-1] * (1 - alpha_slow)
    macd_line = ema_fast - ema_slow
    # Signal line (9-day EMA of MACD)
    alpha_sig = 2 / (signal + 1)
    signal_line = np.copy(macd_line)
    for i in range(1, len(signal_line)):
        signal_line[i] = signal_line[i] * alpha_sig + signal_line[i-1] * (1 - alpha_sig)
    histogram = macd_line - signal_line
    return (
        round(float(macd_line[-1]), 4),
        round(float(signal_line[-1]), 4),
        round(float(histogram[-1]), 4)
    )


def calc_bollinger(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    if len(prices) < period:
        return None, None, None
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return round(float(upper), 2), round(float(sma), 2), round(float(lower), 2)


def get_technical_analysis(ticker_str):
    """Get full technical analysis for a stock."""
    try:
        stock = yf.Ticker(ticker_str)
        # Get 1 year of daily data for indicators
        hist = stock.history(period="1y")
        if hist.empty or len(hist) < 50:
            return None

        closes = hist['Close'].values
        volumes = hist['Volume'].values
        current_price = float(closes[-1])

        ta = {}

        # Moving Averages
        for period, label in [(20, "MA20"), (50, "MA50"), (200, "MA200")]:
            if len(closes) >= period:
                ma = float(np.mean(closes[-period:]))
                ta[label] = {
                    "value": round(ma, 2),
                    "signal": "bullish" if current_price > ma else "bearish",
                    "pct_diff": round((current_price - ma) / ma * 100, 2)
                }

        # RSI
        rsi = calc_rsi(closes)
        ta["rsi"] = {
            "value": rsi,
            "signal": "oversold" if rsi and rsi < 30 else ("overbought" if rsi and rsi > 70 else "neutral"),
            "interpretation": "超賣 (可能反彈)" if rsi and rsi < 30 else ("超買 (可能回調)" if rsi and rsi > 70 else "中性區域")
        }

        # MACD
        macd, signal, hist_val = calc_macd(closes)
        ta["macd"] = {
            "macd_line": macd,
            "signal_line": signal,
            "histogram": hist_val,
            "signal": "bullish" if hist_val and hist_val > 0 else "bearish",
            "interpretation": "MACD 在信號線之上 (看好)" if hist_val and hist_val > 0 else "MACD 在信號線之下 (看淡)"
        }

        # Bollinger Bands
        upper, mid, lower = calc_bollinger(closes)
        if upper and lower:
            bb_position = (current_price - lower) / (upper - lower) * 100 if upper != lower else 50
            ta["bollinger"] = {
                "upper": upper,
                "middle": mid,
                "lower": lower,
                "position_pct": round(bb_position, 1),
                "signal": "oversold_near_lower" if bb_position < 20 else ("overbought_near_upper" if bb_position > 80 else "neutral"),
                "interpretation": "接近布林帶下軌 (超賣)" if bb_position < 20 else ("接近布林帶上軌 (超買)" if bb_position > 80 else "布林帶中軌附近")
            }

        # Volume Analysis
        if len(volumes) >= 20:
            avg_vol = float(np.mean(volumes[-20:]))
            recent_vol = float(np.mean(volumes[-5:]))
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
            ta["volume"] = {
                "avg_20d": int(avg_vol),
                "recent_5d_avg": int(recent_vol),
                "ratio": round(vol_ratio, 2),
                "signal": "high_volume" if vol_ratio > 1.3 else ("low_volume" if vol_ratio < 0.7 else "normal"),
                "interpretation": "成交量高於平均 (活躍)" if vol_ratio > 1.3 else ("成交量低於平均 (淡靜)" if vol_ratio < 0.7 else "成交量正常")
            }

        # Support/Resistance (simple: recent lows/highs)
        if len(closes) >= 50:
            ta["support"] = round(float(np.min(closes[-50:])), 2)
            ta["resistance"] = round(float(np.max(closes[-20:])), 2)
            ta["support_distance"] = round((current_price - ta["support"]) / ta["support"] * 100, 2)
            ta["resistance_distance"] = round((ta["resistance"] - current_price) / current_price * 100, 2)

        # Overall technical signal
        bullish_signals = 0
        bearish_signals = 0
        for key in ["MA20", "MA50"]:
            if key in ta and ta[key]["signal"] == "bullish":
                bullish_signals += 1
            elif key in ta:
                bearish_signals += 1
        if rsi:
            if rsi < 30:
                bullish_signals += 1  # Oversold = potential bounce
            elif rsi > 70:
                bearish_signals += 1
            elif rsi > 50:
                bullish_signals += 1
            else:
                bearish_signals += 1
        if hist_val:
            if hist_val > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1

        ta["overall"] = {
            "bullish": bullish_signals,
            "bearish": bearish_signals,
            "score": round(bullish_signals / max(bullish_signals + bearish_signals, 1) * 10, 1),
            "signal": "strong_bullish" if bullish_signals >= 4 else ("bullish" if bullish_signals >= 3 else ("neutral" if bullish_signals >= 2 else "bearish"))
        }

        return ta
    except Exception as e:
        print(f"TA error for {ticker_str}: {e}")
        return None


# ─── OPTIONS FLOW ───

def get_options_flow(ticker_str):
    """Get options flow data."""
    try:
        stock = yf.Ticker(ticker_str)
        # Get available expiration dates
        expirations = stock.options
        if not expirations or len(expirations) == 0:
            return None

        # Get nearest expiration
        nearest = expirations[0]
        opt = stock.option_chain(nearest)

        calls = opt.calls
        puts = opt.puts

        # Safe int conversion helper
        def safe_int_opt(val):
            try:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return 0
                return int(float(val))
            except:
                return 0

        # Calculate key metrics
        total_call_volume = safe_int_opt(calls['volume'].sum()) if 'volume' in calls.columns else 0
        total_put_volume = safe_int_opt(puts['volume'].sum()) if 'volume' in puts.columns else 0
        total_call_oi = safe_int_opt(calls['openInterest'].sum()) if 'openInterest' in calls.columns else 0
        total_put_oi = safe_int_opt(puts['openInterest'].sum()) if 'openInterest' in puts.columns else 0

        put_call_vol = round(total_put_volume / total_call_volume, 2) if total_call_volume > 0 else None
        put_call_oi = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else None

        # Find unusual volume options (volume > 5x OI)
        unusual = []
        for _, row in calls.iterrows():
            vol = safe_int_opt(row.get('volume'))
            oi = safe_int_opt(row.get('openInterest'))
            if vol > 100 and oi > 0 and vol / oi > 3:
                unusual.append({
                    "type": "CALL",
                    "strike": float(row['strike']),
                    "volume": vol,
                    "openInterest": oi,
                    "lastPrice": float(row.get('lastPrice', 0) or 0),
                    "impliedVolatility": round(float(row.get('impliedVolatility', 0) or 0) * 100, 2),
                })
        for _, row in puts.iterrows():
            vol = safe_int_opt(row.get('volume'))
            oi = safe_int_opt(row.get('openInterest'))
            if vol > 100 and oi > 0 and vol / oi > 3:
                unusual.append({
                    "type": "PUT",
                    "strike": float(row['strike']),
                    "volume": vol,
                    "openInterest": oi,
                    "lastPrice": float(row.get('lastPrice', 0) or 0),
                    "impliedVolatility": round(float(row.get('impliedVolatility', 0) or 0) * 100, 2),
                })

        # Sort by volume descending
        unusual.sort(key=lambda x: x['volume'], reverse=True)
        unusual = unusual[:10]

        # Determine sentiment from put/call
        sentiment = "neutral"
        if put_call_vol is not None:
            if put_call_vol < 0.5:
                sentiment = "bullish"  # More calls = bullish
            elif put_call_vol > 1.5:
                sentiment = "bearish"  # More puts = bearish
            elif put_call_vol < 0.8:
                sentiment = "slightly_bullish"
            elif put_call_vol > 1.2:
                sentiment = "slightly_bearish"

        # Get max pain (approximate: strike with highest OI)
        all_strikes = {}
        for _, row in calls.iterrows():
            s = float(row['strike'])
            all_strikes[s] = all_strikes.get(s, 0) + safe_int_opt(row.get('openInterest'))
        for _, row in puts.iterrows():
            s = float(row['strike'])
            all_strikes[s] = all_strikes.get(s, 0) + safe_int_opt(row.get('openInterest'))
        max_pain = max(all_strikes, key=all_strikes.get) if all_strikes else None

        return {
            "expiration": nearest,
            "total_call_volume": total_call_volume,
            "total_put_volume": total_put_volume,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "put_call_ratio_vol": put_call_vol,
            "put_call_ratio_oi": put_call_oi,
            "sentiment": sentiment,
            "max_pain": max_pain,
            "unusual_activity": unusual,
            "available_expirations": len(expirations),
        }
    except Exception as e:
        print(f"Options error for {ticker_str}: {e}")
        return None


# ─── SOCIAL SENTIMENT ───

def get_social_sentiment(ticker_str, stock_data):
    """Generate social sentiment indicators based on available data."""
    # We use a combination of indicators:
    # 1. Short interest (high short = bearish sentiment)
    # 2. Recent price momentum = retail sentiment proxy
    # 3. Volume spike = attention
    # 4. News count = media attention

    signals = []
    score = 5.0  # Start neutral

    info = stock_data or {}

    # Short interest
    short_pct = info.get("short_pct")
    if short_pct is not None:
        if short_pct > 20:
            signals.append({"type": "bearish", "msg": f"淡倉比例極高 ({short_pct}%)，大量投資者看淡，但亦可能引發挾淡倉"})
            score -= 1.5
        elif short_pct > 10:
            signals.append({"type": "slightly_bearish", "msg": f"淡倉比例偏高 ({short_pct}%)，市場有一定看淡情緒"})
            score -= 0.5
        elif short_pct < 3:
            signals.append({"type": "bullish", "msg": f"淡倉比例極低 ({short_pct}%)，市場普遍看好"})
            score += 1

    # Beta = volatility proxy (high beta stocks get more retail attention)
    beta = info.get("beta")
    if beta is not None:
        if beta > 2:
            signals.append({"type": "neutral", "msg": f"高 Beta ({beta})，散戶關注度高，波動性大"})

    # Volume analysis
    avg_vol = info.get("avg_volume", 0)
    cur_vol = info.get("volume", 0)
    if avg_vol and cur_vol and avg_vol > 0:
        vol_ratio = cur_vol / avg_vol
        if vol_ratio > 2:
            signals.append({"type": "attention", "msg": f"成交量係平均嘅 {vol_ratio:.1f}x，市場高度關注"})
            score += 0.5
        elif vol_ratio < 0.3:
            signals.append({"type": "low_attention", "msg": "成交量極低，市場關注度不足"})

    # Analyst consensus as proxy for professional sentiment
    rec = info.get("recommendation", "")
    if "buy" in str(rec).lower():
        score += 1
        signals.append({"type": "bullish", "msg": f"分析師共識: {rec}，專業機構看好"})
    elif "sell" in str(rec).lower():
        score -= 1
        signals.append({"type": "bearish", "msg": f"分析師共識: {rec}，專業機構看淡"})

    # Insider trading sentiment
    insider_count = len(info.get("insider_transactions", []))
    if insider_count > 0:
        buys = sum(1 for t in info.get("insider_transactions", []) if "buy" in str(t.get("transaction", "")).lower() or "purchase" in str(t.get("transaction", "")).lower())
        sells = insider_count - buys
        if buys > sells:
            signals.append({"type": "bullish", "msg": f"內部人淨買入 ({buys}買 vs {sells}賣)，管理層有信心"})
            score += 1
        elif sells > buys:
            signals.append({"type": "bearish", "msg": f"內部人淨賣出 ({sells}賣 vs {buys}買)，需留意"})
            score -= 0.5

    # Recent price momentum (1m)
    change_pct = info.get("change_pct", 0) or 0
    if change_pct > 5:
        signals.append({"type": "momentum_bullish", "msg": f"今日升 {change_pct}%，短線動能強勁"})
        score += 0.5
    elif change_pct < -5:
        signals.append({"type": "momentum_bearish", "msg": f"今日跌 {change_pct}%，短線受壓"})
        score -= 0.5

    # Clamp score
    score = max(1.0, min(10.0, score))

    # Label
    if score >= 7:
        label = "🔥 市場情緒樂觀"
    elif score >= 5.5:
        label = "🙂 輕微看好"
    elif score >= 4.5:
        label = "😐 中性"
    elif score >= 3:
        label = "😟 輕微看淡"
    else:
        label = "❄️ 市場情緒悲觀"

    return {
        "score": round(score, 1),
        "label": label,
        "signals": signals,
        "reddit_link": f"https://www.reddit.com/search/?q={ticker_str}+wallstreetbets",
        "stocktwits_link": f"https://stocktwits.com/symbol/{ticker_str}",
    }


# ─── PRICE ALERTS (Server-side storage) ───

# Simple in-memory storage for alerts (in production, use a database)
price_alerts = []



# ─── BACKGROUND REFRESH ───

# Cache for frequently accessed stocks
stock_cache = {}
CACHE_TTL = 3600  # 1 hour cache

def refresh_cache():
    """Background task to refresh ALL watchlist stocks periodically."""
    time.sleep(10)  # Wait for server to fully start first
    try:
        all_tickers_raw = get_all_tickers()
        all_tickers = []
        for mkt in ["NASDAQ", "SP500", "DOW", "CRYPTO"]:
            all_tickers.extend([t[0] for t in all_tickers_raw.get(mkt, [])])
        all_tickers = list(set(all_tickers))
    except Exception as e:
        print(f"⚠️  Failed to get ticker list: {e}")
        all_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]  # fallback
    
    while True:
        try:
            for ticker in all_tickers:
                try:
                    data = fetch_stock_data(ticker)
                    if data:
                        stock_cache[ticker] = {"data": data, "timestamp": time.time()}
                except Exception:
                    pass
                time.sleep(1)  # Rate limit
            print(f"🔄 Cache refreshed: {len(stock_cache)} stocks")
        except Exception as e:
            print(f"⚠️  Cache refresh error: {e}")
        time.sleep(3600)

def get_cached_stock(ticker):
    """Get stock from cache if fresh, else fetch new."""
    cached = stock_cache.get(ticker.upper())
    if cached and (time.time() - cached["timestamp"]) < CACHE_TTL:
        return cached["data"]
    data = fetch_stock_data(ticker)
    if data:
        stock_cache[ticker.upper()] = {"data": data, "timestamp": time.time()}
    return data

@app.on_event("startup")
async def startup_event():
    """Start background cache refresh on startup."""
    try:
        thread = threading.Thread(target=refresh_cache, daemon=True)
        thread.start()
        print("🚀 Background cache refresh started")
    except Exception as e:
        print(f"⚠️  Startup warning: {e}")

# ─── FASTAPI ROUTES ───

@app.get("/", response_class=HTMLResponse)
async def home():
    """Main page."""
    import pathlib
    html_path = pathlib.Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/stock/{ticker}")
async def get_stock(ticker: str):
    """API endpoint to get stock analysis (with cache)."""
    ticker = ticker.strip().upper()
    data = get_cached_stock(ticker)

    if data is None:
        raise HTTPException(status_code=404, detail=f"無法獲取 {ticker} 嘅數據。請確認股票代號正確。")

    return data


@app.get("/api/search")
async def search(q: str = Query(..., min_length=1)):
    """Search for a stock."""
    data = fetch_stock_data(q.strip())
    if data is None:
        raise HTTPException(status_code=404, detail=f"找不到 '{q}'")
    return data


@app.get("/api/tickers")
async def list_tickers(market: str = Query(default="all")):
    """List all tickers by market."""
    all_data = get_all_tickers()
    if market.upper() == "ALL":
        result = {}
        for mkt, tickers in all_data.items():
            result[mkt] = [{"ticker": t[0], "name": t[1], "sector": t[2], "market": t[3]} for t in tickers]
        return result
    else:
        market_key = {"NASDAQ": "NASDAQ", "SP500": "SP500", "DOW": "DOW", "CRYPTO": "CRYPTO"}.get(market.upper(), "NASDAQ")
        tickers = all_data.get(market_key, [])
        return [{"ticker": t[0], "name": t[1], "sector": t[2], "market": t[3]} for t in tickers]


@app.get("/api/rankings")
async def get_rankings(market: str = Query(default="all", description="Market: all, NASDAQ, SP500, DOW, CRYPTO")):
    """Return tickers sorted by rating descending (cache only, fast)."""
    all_data = get_all_tickers()
    if market.upper() == "ALL":
        tickers = []
        for mkt in ["NASDAQ", "SP500", "DOW", "CRYPTO"]:
            tickers.extend(all_data.get(mkt, []))
        # Deduplicate by ticker
        seen = set()
        deduped = []
        for t in tickers:
            if t[0] not in seen:
                seen.add(t[0])
                deduped.append(t)
        tickers = deduped
    else:
        market_key = {"NASDAQ": "NASDAQ", "SP500": "SP500", "DOW": "DOW", "CRYPTO": "CRYPTO"}.get(market.upper(), "NASDAQ")
        tickers = all_data.get(market_key, all_data.get("NASDAQ", []))
    results = []
    for t in tickers:
        ticker = t[0]
        cached = stock_cache.get(ticker)
        if cached:
            data = cached["data"]
            results.append({
                "ticker": ticker,
                "name": t[1],
                "sector": t[2],
                "price": data.get("price"),
                "rating": data.get("rating", 0),
                "change_pct": data.get("change_pct"),
            })
        else:
            # No cache yet - return with rating 0 (will be filled by background refresh)
            results.append({
                "ticker": ticker,
                "name": t[1],
                "sector": t[2],
                "price": None,
                "rating": 0,
                "change_pct": None,
            })
    
    results.sort(key=lambda x: x["rating"], reverse=True)
    return {"rankings": results, "cached": len([r for r in results if r["rating"] > 0]), "total": len(results)}


@app.get("/api/metrics")
async def get_metrics():
    """Get metric explanations."""
    return METRIC_EXPLANATIONS



@app.get("/api/stock/{ticker}/technical")
async def get_technical(ticker: str):
    """Get technical analysis for a stock."""
    ticker = ticker.strip().upper()
    ta = get_technical_analysis(ticker)
    if ta is None:
        raise HTTPException(status_code=404, detail=f"無法獲取 {ticker} 嘅技術分析數據")
    return ta


@app.get("/api/stock/{ticker}/options")
async def get_options(ticker: str):
    """Get options flow data."""
    ticker = ticker.strip().upper()
    opts = get_options_flow(ticker)
    if opts is None:
        raise HTTPException(status_code=404, detail=f"無法獲取 {ticker} 嘅期權數據")
    return opts


@app.get("/api/stock/{ticker}/sentiment")
async def get_sentiment(ticker: str):
    """Get social sentiment analysis."""
    ticker = ticker.strip().upper()
    # Fetch basic stock data for sentiment calculation
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Add insider transactions
        insider_data = []
        try:
            insider = stock.insider_transactions
            if insider is not None and not insider.empty:
                for _, row in insider.head(10).iterrows():
                    insider_data.append({
                        "transaction": str(row.get("Transaction", row.get(" transaction", ""))),
                    })
        except:
            pass
        info["insider_transactions"] = insider_data
    except:
        info = {}
    sentiment = get_social_sentiment(ticker, info)
    return sentiment


@app.post("/api/alerts")
async def create_alert(ticker: str = Query(...), price: float = Query(...), direction: str = Query("above")):
    """Create a price alert."""
    alert = {
        "id": len(price_alerts) + 1,
        "ticker": ticker.upper(),
        "price": price,
        "direction": direction,
        "created": datetime.now().isoformat(),
        "triggered": False
    }
    price_alerts.append(alert)
    return {"status": "ok", "alert": alert}


@app.get("/api/alerts")
async def list_alerts():
    """List all price alerts."""
    return {"alerts": price_alerts}


@app.delete("/api/alerts/{alert_id}")
async def delete_alert(alert_id: int):
    """Delete a price alert."""
    global price_alerts
    price_alerts = [a for a in price_alerts if a["id"] != alert_id]
    return {"status": "ok"}


@app.get("/api/stock/{ticker}/full")
async def get_full_analysis(ticker: str):
    """Get complete analysis: fundamentals + technical + options + sentiment."""
    ticker = ticker.strip().upper()
    data = get_cached_stock(ticker)
    if data is None:
        raise HTTPException(status_code=404, detail=f"無法獲取 {ticker} 嘅數據")

    # Add technical analysis
    ta = get_technical_analysis(ticker)
    data["technical"] = ta

    # Add options (may fail for some stocks)
    try:
        opts = get_options_flow(ticker)
        data["options"] = opts
    except:
        data["options"] = None

    # Add sentiment
    sentiment = get_social_sentiment(ticker, data)
    data["sentiment"] = sentiment

    return data



@app.get("/health")
async def health():
    return {"status": "ok", "cached": len(stock_cache)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
