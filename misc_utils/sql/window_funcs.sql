-- Aggreate window function
select request_date
        , round(abs(
            distance_to_travel / monetary_cost
            - avg(distance_per_dollar) over (partition by year_month)
        )::decimal, 2) as diff
    from (
    select *
        , to_char(request_date::date, 'YYYY-MM') as year_month
        , distance_to_travel / monetary_cost as distance_per_dollar
        from uber_request_logs
    ) as Q
    order by request_date desc
;


-- rank, dense_rank, row_number
select distinct department, salary
    from (
    select department
            , salary
            , row_number() over (partition by department order by salary desc) as rank_salary
        from (
            select distinct department, salary
            from twitter_employee
        ) as A
    ) as B
    where B.rank_salary <= 3
    order by department asc, salary desc
;


-- ntile (build a equal-frequency histogram/ percentiles)
select policy_num, state, claim_cost, fraud_score
    from (
    select *
            , ntile(100) over (partition by state order by fraud_score desc) as bin
        from fraud_score
    ) as A
    where bin <= 5
;


-- Cumsum + lag (could be used 'lead', which looks forward)
select *, 100. * (cumsum - lag) / lag as growth
    from (
    select *
            , lag(cumsum, 1) over (order by year asc) as lag
        from (
        select *
            , sum(freq) over (order by year asc rows between unbounded preceding and current row) as cumsum
            from (
            select extract(year from host_since) as year, count(*) as freq
                from airbnb_search_details
                group by extract(year from host_since)
            ) as A
        ) as B
    ) as C
    where lag is not null
;


select *, round((100. * cast(freq - lag as float) / lag)::numeric, 2) as growth
    from (
    select *
        , lag(freq, 1) over (order by year asc) as lag
    from (
        select extract(year from host_since) as year, count(*) as freq
            from airbnb_search_details
            group by extract(year from host_since)
        ) as A
    ) as B
    where lag is not null
;



-- using a window alias (variable that keep an expression)
select year_month
        , round((100. * cast(total_value - prev_total_value as float) / prev_total_value)::numeric, 2) as pct_change
    from (
    select *
        , lag(total_value, 1) over (w) as prev_total_value
        from (
        select to_char(created_at, 'YYYY-MM') year_month, sum(value) as total_value
            from sf_transactions
            group by to_char(created_at, 'YYYY-MM')
        ) as A
        window w as (order by year_month asc)
    ) as B
    where B.prev_total_value is not null
;



select C.first_name
        , O.order_details
        , round((100. * cast(order_cost as float) / (sum(order_cost) over (partition by cust_id)))::numeric, 2) as pct
    from orders O
    inner join customers C
    on C.id = O.cust_id
    order by C.first_name asc
;
