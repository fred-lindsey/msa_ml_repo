-- Part 1: SQLite
--Q1
SELECT
    COUNT (DISTINCT(Email))
FROM
    Customer
WHERE
    Email LIKE "%gmail%"
-- ANS: 8

--Q2:
SELECT
    AlbumId, Title
FROM
    Album
WHERE
    Title LIKE "M%"
ORDER BY
    1 ASC

--Q3:
SELECT
    ROUND(AVG(UnitPrice),2) AS avg_unit_price, GenreId
FROM
    Track
GROUP BY
    GenreId
ORDER BY
    2 DESC

--Q4:
SELECT
    BillingCity, round(AVG(Total),2) AS avg_total
FROM
    Invoice
GROUP BY
    BillingCity
HAVING
    round(AVG(Total),2) > 6;

--Q5:
SELECT
    BillingCity, round(AVG(Total),2) AS avg_total_primary
FROM
    Invoice
GROUP BY
    BillingCity
HAVING
    round(AVG(Total),2) > (SELECT
                                round(AVG(Total),2) AS avg_total
                            FROM
                                Invoice);

--Q6:
CREATE TEMP TABLE Cities AS
    SELECT
        DISTINCT City
    FROM
        Customer
UNION
    SELECT
        DISTINCT City
    FROM
        Employee3;

SELECT
    *
FROM
    Cities;

-- Q7:
SELECT
    COUNT(*)
FROM
    Cities;
-- ANS: 55

-- Part 2: Postgres

-- Q8:
SELECT
    b."origin_airport", COUNT(b."flight_number") AS count_flights
FROM
    "Delays".flights as b
GROUP BY
    1
ORDER BY
    2 DESC
LIMIT 1;

-- Q9: which route has the most cancelled flights?
SELECT
    "origin_airport",
    "destination_airport",
    COUNT(CASE WHEN "cancelled" = 1 THEN 1 ELSE 0 END) AS cancel_count
FROM
    "Delays".flights
GROUP BY
    1, 2
ORDER BY
    3 DESC
LIMIT
    1;

-- Q10: which route has the most cancelled flights due to Airline/Carrier?
SELECT
    "origin_airport",
    "destination_airport",
    COUNT(CASE WHEN "cancelled" = 1
                AND "cancellation_reason" = 'A'
                    THEN 1 ELSE 0 END) AS carrrier_cancel_count
FROM
    "Delays".flights
GROUP BY
    1, 2
ORDER BY
    3 DESC
LIMIT
    1;
-- ANS LAX - JFK, 15

-- Q11
SELECT
    "airline", SUM(CAST("departure_delay" AS INT)) AS delay
FROM
    (SELECT
         "airline", "departure_delay"
     FROM
         "Delays".flights
     WHERE
         "departure_delay" <> '')
GROUP BY
    1
ORDER BY
    1 ASC;
