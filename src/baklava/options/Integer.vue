<template>
  <div class="d-sm-flex">
    <div class="ml-1">{{ name }}</div>
    <IntegerInc :index=0
                :value=this.value
                @value-change="updateValue"/>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import IntegerInc from '@/baklava/input/IntegerInc.vue';

@Component({
  components: { IntegerInc },
})
export default class Integer extends Vue {
  @Prop() value!: number;

  @Prop({ type: String }) name!: string;

  @Prop({ type: Object })
  option!: {
    min?: number;
    max?: number;
  };

  updateValue(value: number) {
    let updated: number = value;
    if (this.option.min !== undefined && updated < this.option.min) {
      updated = this.option.min;
    } else if (this.option.max !== undefined && updated > this.option.max) {
      updated = this.option.max;
    }

    this.$emit('input', updated);
  }
}
</script>
