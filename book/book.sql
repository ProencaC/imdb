drop table if exists tb_abt_imdb;
create table tb_abt_imdb as

SELECT t1.title_id,
       case when t1.type = 'tvMovie' then 1 else 0 end as filme_tv,
       t1.is_adult,
       t1.premiered as ano_estreia,
       t1.runtime_minutes as tempo_duracao,
       case when t1.genres = '\N' then 1 else 0 end as genero_nulo,
       count(t2.title) as numero_titulos,
       count(DISTINCT t3.person_id) as qt_crew,
       t4.votes as num_votos,
       t4.rating
       
       
FROM titles as t1

left join akas as t2
on t1.title_id = t2.title_id

left join crew as t3
on t1.title_id = t3.title_id

left join ratings as t4
on t1.title_id = t4.title_id

where t1.type in ('tvMovie','movie')

group by t1.title_id;
