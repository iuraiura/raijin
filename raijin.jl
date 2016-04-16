# MCMC パッケージ
using Mamba

# URI 系処理パッケージ
@everywhere using HTTPClient.HTTPC, JSON, Zlib

# 個人情報設定

#   アクセスサーバー
#     デモ用
@everywhere SERVER = "https://api-fxpractice.oanda.com"
#     本番用
#@everywhere SERVER = "https://api-fxtrade.oanda.com"

# API アカウント名設定
API_NAME = ""

# アクセストークン設定
@everywhere TOKEN = ""

# 為替変動に対応させるため金額のプール設定。使用金額の三分の一程度は指定しておいたほうが良い
POOL = 1000000

# 投入金額再評価回数。チケット数程度。
UNITS_TIME = 10

# 使用通貨ペア設定
#   フォーマット
#（取引通貨ペア名称、最小変動幅（呼値）ケタ数・日本円での大体の金額・使用可？不可？）
@everywhere CURRECY =
    Dict([
        ("AUD_JPY", (3, 80, true)),
        ("AUD_USD", (5, 100, true)),
        ("EUR_AUD", (5, 80, true)),
        ("EUR_CHF", (5, 120, true)),
        ("EUR_GBP", (5, 140, true)),
        ("EUR_JPY", (3, 140, true)),
        ("EUR_USD", (5, 120, true)),
        ("EUR_CAD", (5, 180, true)),
        ("GBP_CHF", (5, 180, true)),
        ("GBP_JPY", (3, 160, true)),
        ("NZD_USD", (5, 80, true)),
        ("USD_CHF", (5, 120, true)),
        ("USD_JPY", (3, 120, true)),
        ("CAD_JPY", (3, 90, true)),
        ("CHF_JPY", (3, 120, true)),
        ("NZD_JPY", (3, 80, true)),
        ("TRY_JPY", (3, 400, false)),
        ("ZAR_JPY", (3, 700, false)),
        ("SGD_JPY", (3, 80, false)),
        ("GBP_USD", (5, 160, true)),
        ("GBP_CAD", (5, 300, true)),
        ("GBP_AUD", (5, 150, true)),
        ("GBP_NZD", (5, 150, true)),
        ("GBP_SGD", (5, 320, false)),
        ("GBP_CHF", (5, 150, true)),
        ("USD_THB", (5, 40, false)),
        ("USD_SGD", (5, 150, false)),
        ("USD_HKD", (5, 10, false)),
        ("USD_CNH", (5, 700, false)),
        ("USD_CAD", (5, 130, true)),
        ("SGD_CHF", (5, 100, false)),
        ("CAD_CHF", (5, 70, true)),
        ("AUD_CHF", (5, 70, true)),
        ("NZD_CHF", (5, 70, true)),
        ("EUR_SGD", (5, 200, false)),
        ("AUD_SGD", (5, 200, false)),
        ("EUR_TRY", (5, 300, false)),
        ("HKD_JPY", (5, 20, false)),
        ("CHF_HKD", (5, 30, false)),
        ("AUD_NZD", (5, 100, true))
        ])

# システム系設定値
# 標準ヘッダー
@everywhere HEADERS = [("Authorization", "Bearer $TOKEN"),
           ("Connection","Keep-Alive"),
           ("Accept-Encoding" ,"gzip, deflate"),
           ("Content-Type" ,"application/x-www-form-urlencoded")]

# 2.5 %
@everywhere P025 = 1
# 25.0 %
@everywhere P250 = 2
# 50.0 %
@everywhere P500 = 3
# 75.0 %
@everywhere P750 = 4
# 97.5 %
@everywhere P975 = 5
# 最小変動幅（呼値）ケタ数
@everywhere PRECISION = 1
# 日本円での大体の金額
@everywhere UNITS = 2
# 使用可不可
@everywhere  USE = 3
# サンプリング単位時間[一日単位で]
GRANULARITY = "D"
# サンプリング数[四半期を考えている。２０営業日＊３＋５営業日]
TIMES = -65
# サンプリング補正値[サンプリング単位時間だけでは最新の値が反映されないため]
GRANULARITY_S = "H1"
TIMES_S = -24
INTERVAL_TIME = 60
# レバレッジ
LECERAGE = 25
# リトライ回数
RETRY = 5

##########################################################################
# MCMC データモデル
#   Mamba の例題そのまま
model = Model(

    y = Stochastic(1,
        (mu, s2) ->  MvNormal(mu, sqrt(s2)),
        false
     ),

    mu = Logical(1,
        (xmat, beta) -> xmat * beta,
        false
     ),

    # Case 1: Multivariate Normal with independence covariance matrix
    beta = Stochastic(1,
      () -> MvNormal(2, sqrt(1000))
    ),

    s2 = Stochastic(
        () -> InverseGamma(0.001, 0.001)
     )

)

# サンプラリング方法
#   Mamba の例題そのまま
## Hybrid No-U-Turn and Slice Sampling Scheme
scheme = [NUTS(:beta),
               Slice(:s2, 3.0)]

## No-U-Turn Sampling Scheme
#scheme = [NUTS([:beta, :s2])]

#
# データ取得共通関数
@everywhere function get_function(uri::AbstractString)
    try
        status = 0
        code = 0
        message = ""
        ret = []

        for i in 1:RETRY
            tr = HTTPC.get("$SERVER/v1/$uri", headers=HEADERS)
            byesdata = haskey(tr.headers, "Content-Encoding") && tr.headers["Content-Encoding"][1] == "gzip" ? decompress(tr.body.data) : tr.body.data
            ubody = bytestring(byesdata)
            ret = JSON.parse(ubody)
            code = haskey(ret, "code") ? ret["code"] : 0
            message = haskey(ret, "message") ? ret["message"] : ""
            status = tr.http_code
            if code != 68
                break
            end
            sleep(0.5 + rand() / 10.0)
        end

        if status >= 400 println("HTTP Status Code : $status")  end
        if code > 0 println("Error Code : $(code) | message : $message")  end
        return status, code, message, ret
    catch ex
        println(typeof(ex))
        showerror(STDOUT, ex)
    end
end


function get_accountsid(name::AbstractString)
    status, code, message, response = get_function("accounts")
    if code > 0
        return code, 0
    end
    for s in response["accounts"]
        if s["accountName"] == name
            return code, s["accountId"]
        end
    end
end

function get_accounts(account_id::Int64)
    status, code, message, response = get_function("accounts/$account_id")
    code, response
end

@everywhere function get_prices(instrument::AbstractString)
    status, code, message, response = get_function("prices?instruments=$instrument")
    pri = haskey(response, "prices") ? response["prices"][1] : 0.0
    center = (pri["ask"] + pri["bid"]) / 2.0
    spread = (pri["bid"] - pri["ask"]) / 2.0
    code, center, spread
end

@everywhere function get_positions(account_id::Int64, instrument::AbstractString)
    status, code, message, response = get_function("accounts/$account_id/positions/$instrument")
    code, response
end

@everywhere function get_trades(account_id::Int64, instrument::AbstractString)
    status, code, message, response = get_function("accounts/$account_id/trades?instrument=$instrument&count=500")
    code, haskey(response, "trades") ? response["trades"] : []
end

@everywhere function get_trades(account_id::Int64)
    status, code, message, response = get_function("accounts/$account_id/trades?count=500")
    code, haskey(response, "trades") ? response["trades"] : []
end

@everywhere function get_orders(account_id::Int64)
    status, code, message, response = get_function("accounts/$account_id/orders?count=500")
    code, haskey(response, "orders") ? response["orders"] : []
end

function get_candles(instrument::AbstractString, granularity::AbstractString="M1", count::Int64=60, point_time::DateTime=now(Dates.UTC))
    time_str = urlencode("$point_time")
    str = "candles?instrument=$instrument&granularity=$granularity&count=$(abs(count))&candleFormat=midpoint"
    if count > 0
        str *= "&start=$time_str"
    else
        str *= "&end=$time_str"
    end
    status, code, message, response = get_function(str)
    code, haskey(response, "candles") ? response["candles"] : []
end

# リクエスト投入
@everywhere function post_function(uri::AbstractString, option::AbstractString, headers::Array{Tuple{ASCIIString,ASCIIString},1})
    try
        status = 0
        code = 0
        message = ""
        ret = []

        for i in 1:RETRY
            tr = HTTPC.post("$SERVER/v1/$uri", option, headers=headers)
            byesdata = haskey(tr.headers, "Content-Encoding") && tr.headers["Content-Encoding"][1] == "gzip" ? decompress(tr.body.data) : tr.body.data
            ubody = bytestring(byesdata)
            ret = JSON.parse(ubody)
            code = haskey(ret, "code") ? ret["code"] : 0
            message = haskey(ret, "message") ? ret["message"] : ""
            status = tr.http_code
            if code != 68
                break
            end
            sleep(0.5 + rand() / 10.0)
        end

        if status >= 400 println("HTTP Status Code : $status")  end
        if code > 0 println("Error Code : $(code) | message : $message") end
        return status, code, message, ret
    catch ex
        println(typeof(ex))
        showerror(STDOUT, ex)
    end
end

@everywhere function change_trades(account_id::Int64, instrument::AbstractString, trade_id::Int64, stop_loss::Float64=0.0, take_profit::Float64=0.0)
    q = CURRECY[instrument][PRECISION]
    option ="stopLoss=$(floor(stop_loss, q))&takeProfit=$(floor(take_profit, q))"
    headers = [HEADERS; ("X-HTTP-Method-Override", "PATCH")]
    status, code, message, response = post_function("accounts/$account_id/trades/$trade_id", option, headers)
end

function orders(account_id::Int64,
                instrument::AbstractString,
                units::Float64,
                side::AbstractString,
                tt::AbstractString,
                expiry::DateTime=now(),
                price::Float64=0.0,
                stoploss::Float64=0.0,
                takeprofit::Float64=0.0,
                trailingstop::Int64=0)
    time_str = urlencode("$(expiry)")
    q = CURRECY[instrument][PRECISION]
    u = Int(round(units / CURRECY[instrument][UNITS])) + 1
    option ="instrument=$instrument&units=$u&side=$side&type=$tt&expiry=$time_str&price=$(floor(price,q))&stopLoss=$(floor(stoploss,q))&takeProfit=$(floor(takeprofit,q))&trailingStop=$(Int(trailingstop))"
    status, code, message, response = post_function("accounts/$account_id/orders", option, HEADERS)
end

##########################################################################
# MCMC
function simulation(instrument::AbstractString)
  try
    j = 0
    cv = []
    tdata = []
    code, cand1 = get_candles(instrument, GRANULARITY, TIMES, now(Dates.UTC))
    if code > 0
        return 0
    end
    for i in cand1
        j += 1
        push!(cv, i["openMid"])
        push!(tdata, j)
        push!(cv, i["highMid"])
        push!(tdata, j)
        push!(cv, i["lowMid"])
        push!(tdata, j)
        push!(cv, i["closeMid"])
        push!(tdata, j)
    end

    if j == 0
        return 0
    end

    code, cand1 = get_candles(instrument, GRANULARITY_S, TIMES_S, now(Dates.UTC))
    if code > 0
        return 0
    end

    for i in cand1
        push!(cv, i["openMid"])
        push!(tdata, j)
        push!(cv, i["highMid"])
        push!(tdata, j)
        push!(cv, i["lowMid"])
        push!(tdata, j)
        push!(cv, i["closeMid"])
        push!(tdata, j)
    end

    data = Dict{Symbol, Any}(
        :y => cv,
        :t => tdata
    )

    data[:N] = length(data[:y])
    #data[:t] = collect(1:data[:N])
    data[:xmat] = [ones(data[:N]) data[:t]]

    inits = [
      Dict{Symbol, Any}(
        :y => data[:y],
        :beta => rand(Normal(0, 1), 2),
        :s2 => rand(Gamma(1, 1))
      )
    for i in 1:3
    ]

    setsamplers!(model, scheme)
    sim = mcmc(model, data, inits, 10000, burnin=2000, thin=2, chains=2)

    ppd = predict(sim, :y)
    p = quantile(ppd)
    pdata = p.value
  catch ex
      println(typeof(ex))
      showerror(STDOUT, ex)
      return 0
  end
end
##########################################################################

@everywhere function move(account_id::Int64, instrument::AbstractString, pdata::Array)
  try
    code, tr = get_trades(account_id, instrument)
    if code > 0 || length(tr) == 0
        return
    end

    pside = haskey(tr[1], "side") ? tr[1]["side"] : ""
    tside = ""

    code, center, spread = get_prices(instrument)
    if code > 0
        return
    end

    std_u = pdata[end,P750] - pdata[end,P500]
    rate = (pdata[1,P500] - pdata[end,P500]) / std_u

    for t in tr
        id = t["id"]
        side = t["side"]
        price = t["price"]
        stop = t["stopLoss"]
        takep = t["takeProfit"]
        tp = takep
        sp = stop

        code, center, spread = get_prices(instrument)
        if code > 0
            return
        end
        if side == "buy"
            if price < pdata[end,P500]
                tp = pdata[end,P500]
            else
                tp = price + std_u / 10.0
            end
            if price < center
                sp = mean([maximum([price, stop]), center])
            else
                sp = pdata[end,P025]
            end
        elseif side == "sell"
            if price > pdata[end,P500]
                tp = pdata[end,P500]
            else
                tp = price - std_u / 10.0
            end
            if price > center
                sp = mean([minimum([price, stop]), center])
            else
                sp = pdata[end,P975]
            end
        end
        #println("$(now()) : レベル調整　side = $side | price = $price | center = $center | sp = $sp | tp = $tp")
        change_trades(account_id, instrument, id, sp, tp)
    end

    code, tr = get_trades(account_id, instrument)
    if code > 0 || length(tr) == 0
        return
    end

    bstop = pside == "buy" ? 0.0 : typemax(Float64)

    for t  in tr
        side = t["side"]
        stop = t["stopLoss"]
        price = t["price"]
        if side =="buy"
            bstop = maximum([bstop, stop])
        elseif side == "sell"
            bstop = minimum([bstop, stop])
        end
    end

    for t in tr
        id = t["id"]
        side = t["side"]
        price = t["price"]
        stop = t["stopLoss"]
        takep = t["takeProfit"]
        tp = takep
        sp = stop

        code, center, spread = get_prices(instrument)
        if code > 0
            return
        end

        if side == "buy"
            if (price < stop < center) && (bstop > stop)
              sleep(0.6)
              code, center, spread = get_prices(instrument)
              if code > 0
                  return
              end
              if (price < stop < center) && (bstop > stop)
                change_trades(account_id, instrument, id, bstop, tp)
              end
            end
        elseif side == "sell"
            if (price > stop > center) && (bstop < stop)
              sleep(0.6)
              code, center, spread = get_prices(instrument)
              if code > 0
                  return
              end
              if  (price > stop > center) && (bstop < stop)
                change_trades(account_id, instrument, id, bstop, tp)
              end
            end
        end
    end
  catch ex
      println(typeof(ex))
      showerror(STDOUT, ex)
      return
  end
end

@everywhere function adjustment_stoploss(account_id::Int64)
  while true
    try
      sleep(3)
      code, tr = get_trades(account_id)
      if code > 0 || length(tr) == 0
          continue
      end
      inst = []

      for t in tr
        push!(inst, t["instrument"])
      end
      pprice = Dict{AbstractString, Float64}()
      for i in unique(inst)
        code, center, spread = get_prices(i)
        if code > 0
            break
        end
        pprice[i] = center
        sleep(1/15)
      end

      for t in tr
        id = t["id"]
        side = t["side"]
        price = t["price"]
        stop = t["stopLoss"]
        takep = t["takeProfit"]
        instrument = t["instrument"]
        tp = takep
        sp = stop

        if side == "buy"
          if price < pprice[instrument]
            code, center, spread = get_prices(instrument)
            if code > 0
                break
            end
            if price < center
              sleep(0.6)
              code, center, spread = get_prices(instrument)
              if code > 0
                  break
              end
              if price < center
                change_trades(account_id, instrument, id, maximum([mean([price, center]), stop]), tp)
              end
            end
          end
        elseif side == "sell"
          if price > pprice[instrument]
            code, center, spread = get_prices(instrument)
            if code > 0
                break
            end
            if price > center
              sleep(0.6)
              code, center, spread = get_prices(instrument)
              if code > 0
                  break
              end
              if price > center
                change_trades(account_id, instrument, id, minimum([mean([price, center]), stop]), tp)
              end
            end
          end
        end
      end

      pstop = Dict{AbstractString, Float64}()
      code, ps = get_positions(account_id, "")
      if code > 0
          continue
      end

      for p in ps["positions"]
          instrument = p["instrument"]
          side = p["side"]
          pstop[instrument] = side == "buy" ? 0.0 : typemax(Float64)
      end

      code, tr = get_trades(account_id)
      if code > 0 || length(tr) == 0
          continue
      end
      for t in tr
        side = t["side"]
        stop = t["stopLoss"]
        instrument = t["instrument"]
        if side == "buy"
          pstop[instrument] = maximum([pstop[instrument], stop])
        elseif side == "sell"
          pstop[instrument] = minimum([pstop[instrument], stop])
        end
      end
      for t in tr
        id = t["id"]
        side = t["side"]
        price = t["price"]
        stop = t["stopLoss"]
        takep = t["takeProfit"]
        instrument = t["instrument"]
        tp = takep
        sp = stop

        if side == "buy"
          if price < pprice[instrument]
            code, center, spread = get_prices(instrument)
            if code > 0
                break
            end
            if (price < center) && (pstop[instrument] < center)
              sleep(0.6)
              code, center, spread = get_prices(instrument)
              if code > 0
                  break
              end
              if (price < center) && (pstop[instrument] < center)
                change_trades(account_id, instrument, id, pstop[instrument], tp)
              end
            end
          end
        elseif side == "sell"
          if price > pprice[instrument]
            code, center, spread = get_prices(instrument)
            if code > 0
                break
            end
            if (price > center) && (pstop[instrument] > center)
              sleep(0.6)
              code, center, spread = get_prices(instrument)
              if code > 0
                  break
              end
              if (price > center) && (pstop[instrument] > center)
                change_trades(account_id, instrument, id, pstop[instrument], tp)
              end
            end
          end
        end
      end
    catch ex
        println(typeof(ex))
        showerror(STDOUT, ex)
    end
  end
end

function positions(account_id::Int64, instrument::AbstractString, side::AbstractString, unit::Float64)

  try

    if !(haskey(CURRECY, instrument))
        println("$(now()) : 通貨ペアが設定されておりません（$instrument）")
        return 0
    end

    println("$(now())：シミュレーション開始")

    @time pdata = simulation(instrument)
    if pdata == 0
        return 0
    end

    cv = []
    code, cand = get_candles(instrument, "M5", -288, now(Dates.UTC))
    if code > 0
        return 0
    end
    for i in cand
        push!(cv, i["openMid"])
        push!(cv, i["highMid"])
        push!(cv, i["lowMid"])
        push!(cv, i["closeMid"])
    end

    code, center, spread = get_prices(instrument)
    if code > 0
      println("$(now())：価格取得失敗（$code）")
      return 0
    end

    std_u = pdata[end,P750] - pdata[end,P500]
    rate = (pdata[end-1,P500] - pdata[end,P500]) / pdata[end,P500]

    tt = ""
    price = 0.0
    takeprofit = pdata[end,P500]
    stoploss = 0.0
    trailingstop = 0.0

    # オーダー継続時間の設定
    next_time = now(Dates.UTC) + Dates.Minute(INTERVAL_TIME)
    before_data = mean(cv)

    if center < pdata[end,P250] && side != "sell" && (pdata[end,P500] < pdata[1,P500]) && before_data < pdata[end,P500]
      @async orders(account_id, instrument, unit, "buy", "stop", next_time, pdata[end,P250], pdata[end,P025], pdata[end,P500])
    elseif pdata[end,P250] < center < pdata[end,P500] && side != "sell" && (pdata[end,P500] < pdata[1,P500]) && before_data < pdata[end,P500]
      @async orders(account_id, instrument, unit, "buy", "limit", next_time, pdata[end,P250], pdata[end,P025], pdata[end,P500])
    elseif center > pdata[end,P750] && side != "buy" && (pdata[end,P500] > pdata[1,P500]) && before_data > pdata[end,P500]
      @async orders(account_id, instrument, unit, "sell", "stop", next_time, pdata[end,P750], pdata[end,P975], pdata[end,P500])
    elseif pdata[end,P750] > center > pdata[end,P500] && side != "buy" && (pdata[end,P500] > pdata[1,P500]) && before_data > pdata[end,P500]
      @async orders(account_id, instrument, unit, "sell", "limit", next_time, pdata[end,P750], pdata[end,P975], pdata[end,P500])
    end

    println("$(now()) : シミュレーション結果：通貨ペア（$instrument）現在値=$center")
    println("$(now()) : 97.5%[短期]=$(pdata[end,P975])|[長期]=$(pdata[1,P975])")
    println("$(now()) : 75.0%[短期]=$(pdata[end,P750])|[長期]=$(pdata[1,P750])")
    println("$(now()) : 50.0%[短期]=$(pdata[end,P500])|[長期]=$(pdata[1,P500])")
    println("$(now()) : 25.0%[短期]=$(pdata[end,P250])|[長期]=$(pdata[1,P250])")
    println("$(now()) :  2.5%[短期]=$(pdata[end,P025])|[長期]=$(pdata[1,P025])")

    @async move(account_id, instrument, pdata)
  catch ex
      println(typeof(ex))
      showerror(STDOUT, ex)
      return 0
  end
end

# main function

n = 0
unit = 0
last_time = now(Dates.UTC) - Dates.Second(30)
code, account_id = get_accountsid(API_NAME)

cur = []
for c in keys(CURRECY)
    if CURRECY[c][USE] push!(cur, c) end
end

println("$(now()) : アカウントID取得：$account_id")

if code > 0
    println("$(now()) : アカウントID取得失敗")
    exit
end

@async adjustment_stoploss(account_id)

while true
  try
    println("******************************************")
    println("$(now()) : ループ No. $n")

    if mod(n, UNITS_TIME) == 0 || unit < 1.0
        code, account_info = get_accounts(account_id)
        marg = haskey(account_info, "marginAvail") ? account_info["marginAvail"] : 0
        unit = ((marg - POOL) * LECERAGE + 0.0001) / UNITS_TIME
        if unit < 0.0
            unit = 0.001
        end
        println("$(now()) : 注文単位（￥） $unit")
    end

    n += 1

    println("＊＊＊＊＊　保有ポジション処理（開始）＊＊＊＊＊")
    code, ps = get_positions(account_id, "")
    if code == 0 && length(ps) > 0
      for p in ps["positions"]
          instrument = p["instrument"]
          side = p["side"]
          avgprice = p["avgPrice"]
          println("")
          println("ーーーーー　通貨ペア：$instrument　ーーーーー")
          flush(STDOUT)
          if in(instrument, cur)
              positions(account_id, instrument, side, unit)
          end
      end
    end

    println("＊＊＊＊＊　保有ポジション処理（終了）＊＊＊＊＊")
    println("")
    println("")
    flush(STDOUT)

    println("＊＊＊＊＊　注文中ポジション処理（開始）＊＊＊＊＊")
    code, tr = get_orders(account_id)
    if code == 0 || length(tr) > 0
      inst = []

      for t in tr
        push!(inst, t["instrument"])
      end
      for i in unique(inst)
        println("ーーーーー　通貨ペア：$i　ーーーーー")
        flush(STDOUT)
        side = ""
        for t in tr
            if i == t["instrument"]
                side = t["side"]
            end
        end
        positions(account_id, i, side, unit)
      end
      println("＊＊＊＊＊　注文中ポジション処理（終了）＊＊＊＊＊")
      println("")
      println("")
      flush(STDOUT)
    end

    println("＊＊＊＊＊　ランダム選択処理（開始）＊＊＊＊＊")
    instrument = cur[rand(1:length(cur))]
    println("ーーーーー　通貨ペア：$instrument　ーーーーー")
    side = ""
    for p in ps["positions"]
        if instrument == p["instrument"]
            side = p["side"]
        end
    end
    positions(account_id, instrument, side, unit)

    last_time = now(Dates.UTC)
    println("$(now()) : End")
    println("******************************************")
    flush(STDOUT)
  catch ex
      println(typeof(ex))
      showerror(STDOUT, ex)
  end
end
